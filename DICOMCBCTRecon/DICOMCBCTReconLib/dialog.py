"""Reconstruction window shown when a rotational XA run is loaded from the DICOM browser.

    top left     raw projections in cine (frame number and gantry angle)
    other panels axial, coronal and sagittal slices of the reconstruction
    side         projection preprocessing, geometry corrections, method (FDK / FDK + SART / SART),
                 filter, volume grid and output units

Reconstruct runs with the settings on screen; Load is enabled only when the preview matches them.
Layout and widget helpers follow the SPECT reconstruction window of DICOMSpectReconPlugin.
"""

import logging
import math
import time
import traceback

import numpy as np
import qt
import slicer

from . import display
from . import geometry as geom
from . import reconstruction as recon
from . import xadata

METHOD_FDK = "FDK (filtered backprojection)"
METHOD_FDK_SART = "FDK + SART refinement"
METHOD_SART = "SART (iterative only)"
METHODS = [METHOD_FDK, METHOD_FDK_SART, METHOD_SART]
# SART (iterations, subsets, relaxation): refining FDK needs only small corrections, starting from an
# empty volume needs bigger and more steps to bring in fine detail (4 mm rod contrast on the synthetic
# phantom: 46% with 3 x 12 at 0.5, 95% with 3 x 24 at 1.0)
SART_DEFAULTS = {METHOD_FDK_SART: (3, 12, 0.5), METHOD_SART: (3, 24, 1.0)}

UNITS_HU_AUTO = "HU, water from the soft-tissue peak"
UNITS_HU = "HU, fixed water value"
UNITS_MU = "Attenuation (1/cm)"
UNITS = [UNITS_HU_AUTO, UNITS_HU, UNITS_MU]
HU_UNITS = (UNITS_HU_AUTO, UNITS_HU)

MATRIX_SIZES = [128, 192, 256, 384, 512]
BINNINGS = [1, 2, 4]

BUTTON_BLUE = "#1f6fd1"
BUTTON_GREEN = "#1b6e2d"


# ---------------------------------------------------------------------------
# Widget helpers (as in the SPECT window)
# ---------------------------------------------------------------------------
def _group(title):
    box = qt.QGroupBox(title)
    grid = qt.QGridLayout(box)
    grid.setContentsMargins(6, 4, 6, 4)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(3)
    return box, grid


def _colorButton(button, color):
    button.setMinimumHeight(30)
    button.setStyleSheet(
        f"QPushButton {{ background-color: {color}; color: white; font-weight: bold; border: none; "
        f"border-radius: 4px; padding: 4px 10px; }}"
        f"QPushButton:hover {{ border: 2px solid rgba(255, 255, 255, 120); }}"
        f"QPushButton:disabled {{ background-color: #9a9a9a; color: #e8e8e8; }}")


def _combo(items):
    combo = qt.QComboBox()
    combo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(8)
    for item in items:
        combo.addItem(str(item))
    return combo


def _doubleSpin(minimum, maximum, value, decimals=2, step=0.1, suffix=""):
    spin = qt.QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _intSpin(minimum, maximum, value, suffix=""):
    spin = qt.QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    return spin


class WheelScrollFilter(qt.QObject):
    """Forwards mouse wheel events over a preview panel to a slider."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.Wheel:
            try:
                delta = event.angleDelta().y()
            except AttributeError:
                delta = event.delta()
            if delta != 0:
                self.callback(1 if delta > 0 else -1)
            return True
        return False


class BusyDialogGuard(qt.QObject):
    """While a reconstruction runs, Escape and the close button stop it instead of closing."""

    def __init__(self, isBusy, requestStop):
        super().__init__()
        self.isBusy = isBusy
        self.requestStop = requestStop

    def eventFilter(self, obj, event):
        if not self.isBusy():
            return False
        if event.type() == qt.QEvent.Close:
            event.ignore()
            self.requestStop()
            return True
        if event.type() == qt.QEvent.KeyPress and event.key() == qt.Qt.Key_Escape:
            self.requestStop()
            return True
        return False


# ---------------------------------------------------------------------------
# Reconstruction window
# ---------------------------------------------------------------------------
class CbctReconDialog:
    MIN_VIEW_SIZE = 180
    MAX_VIEW_SIZE = 440
    SIDE_PANEL_WIDTH = 430

    def __init__(self, inputDcm, title, otherRuns=None, sweepIndex=0):
        self.pendingLog = []                    # messages logged before the window exists
        self.inputDcm = inputDcm
        self.sweepIndex = sweepIndex
        # [{"file", "sweep", "label"}] sweeps (of this or other runs) with the same trajectory
        self.otherRuns = list(otherRuns or [])
        self.header = xadata.readHeader(inputDcm)
        self.runGeometry = xadata.readGeometry(self.header, log=self.log)
        self.geometry = xadata.sweepGeometry(self.runGeometry, sweepIndex)
        self.allFrames = None                   # every frame of the file (FOV transform undone)
        self.frames = None                      # frames of the sweep being reconstructed
        self.otherFrames = {}                   # (file, sweep) -> frames of a sweep to subtract
        self.integrals = None
        self.integralsKey = None
        self.reconArray = None                  # last reconstruction, output units, [nz, ny, nx]
        self.reconWindow = (0.0, 1.0)
        self.reconGrid = None
        self.image = None                       # SimpleITK image handed to Slicer
        self.info = {}
        self.appliedParams = None
        self.busy = False
        self.abortRequested = False
        self.lastRunFailed = False
        self.projCineTimer = None
        self.wheelFilters = []
        self.viewSize = self._computeViewSize()
        self.renderer = display.PixmapRenderer(self.viewSize)
        self._buildUI(title)

    # ---------------- sizing ----------------
    def _availableScreenSize(self):
        try:
            geometry = slicer.util.mainWindow().screen().availableGeometry()
            return geometry.width(), geometry.height()
        except Exception:
            pass
        try:
            geometry = qt.QApplication.desktop().availableGeometry()
            return geometry.width(), geometry.height()
        except Exception:
            return 1280, 800

    def _computeViewSize(self):
        screenWidth, screenHeight = self._availableScreenSize()
        byHeight = (screenHeight - 140 - 2 * 80) / 2
        byWidth = (screenWidth - self.SIDE_PANEL_WIDTH - 90) / 2
        return int(max(self.MIN_VIEW_SIZE, min(self.MAX_VIEW_SIZE, min(byHeight, byWidth))))

    # ---------------- UI ----------------
    def _makePanel(self, gridLayout, title, row, column):
        container = qt.QWidget()
        box = qt.QVBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        header = qt.QLabel(f"<b>{title}</b>")
        header.setAlignment(qt.Qt.AlignCenter)
        view = qt.QLabel()
        view.setFixedSize(self.viewSize, self.viewSize)
        view.setAlignment(qt.Qt.AlignCenter)
        view.setStyleSheet("background-color: black; color: gray;")
        slider = qt.QSlider(qt.Qt.Horizontal)
        slider.setEnabled(False)
        info = qt.QLabel("")
        info.setAlignment(qt.Qt.AlignCenter)
        for widget in (header, view, slider, info):
            box.addWidget(widget)
        gridLayout.addWidget(container, row, column)
        self._installWheelScroll(view, slider)
        return view, slider, info

    def _installWheelScroll(self, view, slider):
        def step(direction):
            if not slider.isEnabled():
                return
            slider.setValue(max(slider.minimum, min(slider.maximum, slider.value + direction)))
        try:
            wheelFilter = WheelScrollFilter(step)
            view.installEventFilter(wheelFilter)
            self.wheelFilters.append(wheelFilter)
        except Exception:
            logging.warning("Mouse wheel scrolling not available:\n" + traceback.format_exc())

    def _buildUI(self, title):
        self.dialog = qt.QDialog(slicer.util.mainWindow())
        self.dialog.setWindowTitle(f"Cone-beam CT reconstruction - {title}")
        self.dialog.setModal(True)
        self.busyGuard = BusyDialogGuard(lambda: self.busy, self.requestStop)
        self.dialog.installEventFilter(self.busyGuard)
        outer = qt.QHBoxLayout(self.dialog)

        # --- previews ---
        views = qt.QGridLayout()
        outer.addLayout(views)
        self.projView, self.projSlider, self.projInfo = self._makePanel(views, "Projections", 0, 0)
        self.axialView, self.axialSlider, self.axialInfo = self._makePanel(views, "Axial", 0, 1)
        self.coronalView, self.coronalSlider, self.coronalInfo = self._makePanel(views, "Coronal", 1, 0)
        self.sagittalView, self.sagittalSlider, self.sagittalInfo = self._makePanel(views, "Sagittal", 1, 1)
        self.projSlider.connect("valueChanged(int)", self.updateProjView)
        self.axialSlider.connect("valueChanged(int)", lambda i: self.updateReconViews())
        self.coronalSlider.connect("valueChanged(int)", lambda i: self.updateReconViews())
        self.sagittalSlider.connect("valueChanged(int)", lambda i: self.updateReconViews())

        # --- side panel ---
        sideWidget = qt.QWidget()
        side = qt.QVBoxLayout(sideWidget)
        side.setContentsMargins(4, 0, 4, 0)
        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sideWidget)
        scroll.setMinimumWidth(self.SIDE_PANEL_WIDTH)
        scroll.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        summary = qt.QLabel(xadata.describeGeometry(self.geometry))
        summary.setWordWrap(True)
        side.addWidget(summary)

        cine = qt.QHBoxLayout()
        self.cineCheckBox = qt.QCheckBox("Play projections")
        self.cineCheckBox.connect("toggled(bool)", self.onCineToggled)
        self.fpsSpin = _intSpin(1, 60, 25, " fps")
        self.fpsSpin.connect("valueChanged(int)", self.onFpsChanged)
        cine.addWidget(self.cineCheckBox)
        cine.addWidget(self.fpsSpin)
        side.addLayout(cine)

        # Projections
        box, grid = _group("Projections")
        self.intensityCombo = _combo(recon.INTENSITY_MODES)
        self.intensityCombo.setCurrentIndex(recon.INTENSITY_MODES.index(recon.defaultIntensityMode(self.geometry)))
        self.intensityCombo.setToolTip(
            "How pixel values relate to the detected X-ray intensity (set from PixelIntensityRelationship). "
            "GE stores raw rotational frames square-root encoded. Line integrals are ln(I0/I).")
        self.i0Combo = _combo(recon.I0_MODES)
        self.i0Combo.setToolTip("Unattenuated intensity I0: estimated from the brightest pixels "
                                f"({recon.I0_PERCENTILE} percentile), per run or per frame, or a fixed value.")
        maxValue = float(2 ** int(self.geometry.get("bitsStored") or 16))
        self.i0Spin = _doubleSpin(1.0, max(maxValue, 65535.0), maxValue - 1, 0, 10.0)
        self.binningCombo = _combo([f"{b} x {b}" for b in BINNINGS])
        self.binningCombo.setToolTip("Average neighbouring detector pixels: faster, less noise, less detail.")
        self.subtractCombo = _combo(["None"] + [run["label"] for run in self.otherRuns])
        self.subtractCombo.setToolTip(
            "Subtract the line integrals of another run with the same trajectory (3D DSA): "
            "contrast-filled run minus native (mask) run leaves the contrast only.")
        self.subtractCombo.setEnabled(bool(self.otherRuns))
        grid.addWidget(qt.QLabel("Intensity:"), 0, 0)
        grid.addWidget(self.intensityCombo, 0, 1, 1, 2)
        grid.addWidget(qt.QLabel("I0:"), 1, 0)
        grid.addWidget(self.i0Combo, 1, 1)
        grid.addWidget(self.i0Spin, 1, 2)
        grid.addWidget(qt.QLabel("Binning:"), 2, 0)
        grid.addWidget(self.binningCombo, 2, 1)
        grid.addWidget(qt.QLabel("Subtract run:"), 3, 0)
        grid.addWidget(self.subtractCombo, 3, 1, 1, 2)
        self.i0Combo.connect("currentIndexChanged(int)", self._updateI0State)
        side.addWidget(box)

        # Geometry corrections
        box, grid = _group("Geometry corrections")
        self.flipColumnsCheckBox = qt.QCheckBox("Mirror columns")
        self.flipColumnsCheckBox.setToolTip("Use when the reconstruction is blurred/doubled everywhere: "
                                            "the stored images are mirrored relative to the rotation.")
        self.flipRowsCheckBox = qt.QCheckBox("Mirror rows")
        self.flipRowsCheckBox.setToolTip("Use when head and feet are swapped in the coronal view.")
        self.negateCheckBox = qt.QCheckBox("Negate angles")
        self.negateCheckBox.setToolTip("Use when the volume is a left-right mirror image of the patient.")
        self.angleOffsetSpin = _doubleSpin(-180.0, 180.0, 0.0, 1, 0.5, " deg")
        self.offsetUSpin = _doubleSpin(-50.0, 50.0, 0.0, 2, 0.1, " mm")
        self.offsetUSpin.setToolTip("Detector shift along the image columns (at the detector). Tune it when "
                                    "small high-contrast objects look like rings or arcs.")
        self.offsetVSpin = _doubleSpin(-50.0, 50.0, 0.0, 2, 0.1, " mm")
        grid.addWidget(self.flipColumnsCheckBox, 0, 0)
        grid.addWidget(self.flipRowsCheckBox, 0, 1)
        grid.addWidget(self.negateCheckBox, 0, 2)
        self.calibrationCheckBox = qt.QCheckBox("Use the GE 3D calibration stored in the run")
        hasCalibration = self.geometry.get("calibration") is not None
        self.calibrationCheckBox.setChecked(hasCalibration)
        self.calibrationCheckBox.setEnabled(hasCalibration)
        self.calibrationCheckBox.setToolTip(
            "Calibrated projection matrices (private GEMS_XR3DCAL_01) describe the real C-arm motion "
            "(sag, detector shift) and give sharper images than the nominal angles."
            if hasCalibration else "This run carries no GE 3D calibration.")
        grid.addWidget(self.calibrationCheckBox, 3, 0, 1, 3)
        grid.addWidget(qt.QLabel("Angle offset:"), 1, 0)
        grid.addWidget(self.angleOffsetSpin, 1, 1)
        grid.addWidget(qt.QLabel("Detector shift u / v:"), 2, 0)
        grid.addWidget(self.offsetUSpin, 2, 1)
        grid.addWidget(self.offsetVSpin, 2, 2)
        side.addWidget(box)

        # Reconstruction
        box, grid = _group("Reconstruction")
        self.methodCombo = _combo(METHODS)
        self.filterCombo = _combo(recon.FILTERS)
        self.filterCombo.setCurrentIndex(recon.FILTERS.index(recon.DEFAULT_FILTER))
        self.cutoffSpin = _doubleSpin(0.1, 1.0, 0.7, 2, 0.05)
        self.cutoffSpin.setToolTip("Filter cut-off as a fraction of the Nyquist frequency (lower = smoother).")
        self.cutoffVSpin = _doubleSpin(0.1, 1.0, 0.7, 2, 0.05)
        self.cutoffVSpin.setToolTip("Hann window along the detector columns (vertical), as a fraction of the "
                                    "Nyquist frequency: smooths between slices. Follows the cut-off until changed.")
        self.lastCutoff = 0.7  # the vertical cut-off follows the cut-off while they are equal
        self.truncationSpin = _intSpin(0, 100, 25, " %")
        self.truncationSpin.setToolTip("Lateral truncation correction: each detector row is extended by this "
                                       "fraction of its width with its mirror image about the edge value, "
                                       "fading to zero. 0 switches it off.")
        self.itersSpin = _intSpin(1, 50, 3)
        self.subsetsSpin = _intSpin(1, 100, 12)
        self.relaxSpin = _doubleSpin(0.05, 1.5, 0.5, 2, 0.05)
        self.itersSpin.setToolTip("More iterations bring in finer detail, and more noise.")
        self.subsetsSpin.setToolTip("Updates per iteration (each uses every n-th view): more subsets converge faster.")
        self.relaxSpin.setToolTip("Step size of each update: lower is smoother and slower to converge.")
        self.cpuCheckBox = qt.QCheckBox("Use the CPU even when a GPU is available")
        self.cpuCheckBox.setToolTip("A GPU without enough memory for the volume can be slower than the CPU "
                                    "(the driver then swaps to system memory).")
        self.lastIterativeMethod = METHOD_FDK_SART  # whose defaults the SART spin boxes hold
        grid.addWidget(qt.QLabel("Method:"), 0, 0)
        grid.addWidget(self.methodCombo, 0, 1, 1, 3)
        grid.addWidget(qt.QLabel("Filter:"), 1, 0)
        grid.addWidget(self.filterCombo, 1, 1)
        grid.addWidget(qt.QLabel("Cut-off:"), 1, 2)
        grid.addWidget(self.cutoffSpin, 1, 3)
        grid.addWidget(qt.QLabel("Truncation:"), 2, 0)
        grid.addWidget(self.truncationSpin, 2, 1)
        grid.addWidget(qt.QLabel("Vertical cut-off:"), 2, 2)
        grid.addWidget(self.cutoffVSpin, 2, 3)
        grid.addWidget(qt.QLabel("SART iterations:"), 3, 0)
        grid.addWidget(self.itersSpin, 3, 1)
        grid.addWidget(qt.QLabel("Subsets:"), 3, 2)
        grid.addWidget(self.subsetsSpin, 3, 3)
        grid.addWidget(qt.QLabel("Relaxation:"), 4, 0)
        grid.addWidget(self.relaxSpin, 4, 1)
        grid.addWidget(self.cpuCheckBox, 5, 0, 1, 4)
        self.methodCombo.connect("currentIndexChanged(int)", self._updateMethodState)
        self.cutoffSpin.connect("valueChanged(double)", self._updateCutoffV)
        side.addWidget(box)

        # Volume and units
        box, grid = _group("Volume")
        self.matrixCombo = _combo(MATRIX_SIZES)
        self.matrixCombo.setCurrentIndex(MATRIX_SIZES.index(256))
        self.voxelSpin = _doubleSpin(0.0, 5.0, 0.0, 2, 0.05, " mm")
        self.voxelSpin.setSpecialValueText("auto (fit FOV)")
        self.slicesSpin = _intSpin(0, 1024, 0)
        self.slicesSpin.setSpecialValueText("auto")
        self.unitsCombo = _combo(UNITS)
        self.muWaterSpin = _doubleSpin(0.05, 1.0, 0.20, 3, 0.005, " /cm")
        self.muWaterSpin.setToolTip("Attenuation of water at the effective energy, for the HU scale. Values from "
                                    "display-processed frames are relative, so set this so that soft tissue reads ~40 HU.")
        grid.addWidget(qt.QLabel("Matrix:"), 0, 0)
        grid.addWidget(self.matrixCombo, 0, 1)
        grid.addWidget(qt.QLabel("Voxel:"), 0, 2)
        grid.addWidget(self.voxelSpin, 0, 3)
        grid.addWidget(qt.QLabel("Slices:"), 1, 0)
        grid.addWidget(self.slicesSpin, 1, 1)
        grid.addWidget(qt.QLabel("Units:"), 2, 0)
        grid.addWidget(self.unitsCombo, 2, 1)
        grid.addWidget(qt.QLabel("Water:"), 2, 2)
        grid.addWidget(self.muWaterSpin, 2, 3)
        self.cropCheckBox = qt.QCheckBox("Crop to the field of view seen by every view")
        self.cropCheckBox.setChecked(True)
        self.cropCheckBox.setToolTip("Outside this cylinder some views miss the voxel, so values there are "
                                     "incomplete (streaks, wrong levels). Untick to keep the whole square grid; "
                                     "set a larger voxel size to reconstruct a wider area.")
        grid.addWidget(self.cropCheckBox, 3, 0, 1, 4)
        self.unitsCombo.connect("currentIndexChanged(int)", self._updateUnitsState)
        side.addWidget(box)

        # Actions
        self.applyButton = qt.QPushButton("Reconstruct")
        _colorButton(self.applyButton, BUTTON_BLUE)
        self.applyButton.connect("clicked()", self.onApply)
        self.loadButton = qt.QPushButton("Load")
        _colorButton(self.loadButton, BUTTON_GREEN)
        self.loadButton.connect("clicked()", self.onLoad)
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.connect("clicked()", self.onCancelClicked)
        buttons = qt.QHBoxLayout()
        buttons.addWidget(self.applyButton)
        buttons.addWidget(self.loadButton)
        buttons.addWidget(self.cancelButton)
        side.addLayout(buttons)
        self.staleNote = qt.QLabel("")
        self.staleNote.setWordWrap(True)
        side.addWidget(self.staleNote)
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 1)
        self.progressBar.setValue(0)
        side.addWidget(self.progressBar)
        self.progressLabel = qt.QLabel("")
        side.addWidget(self.progressLabel)
        self.logText = qt.QPlainTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setMinimumHeight(140)
        side.addWidget(self.logText, 1)

        for line in self.pendingLog:
            self.logText.appendPlainText(line)
        self.pendingLog = []
        self._updateI0State()
        self._updateMethodState()
        self._updateUnitsState()
        self._watchSettings()
        self._updateLoadState()

    def _updateI0State(self, *args):
        self.i0Spin.setEnabled(self.i0Combo.currentText == recon.I0_MANUAL)

    def _updateMethodState(self, *args):
        iterative = self.methodCombo.currentText != METHOD_FDK
        for widget in (self.itersSpin, self.subsetsSpin, self.relaxSpin):
            widget.setEnabled(iterative)
        filtered = self.methodCombo.currentText != METHOD_SART
        for widget in (self.filterCombo, self.cutoffSpin, self.cutoffVSpin):
            widget.setEnabled(filtered)
        # Switch the SART settings to the new method's defaults unless the user has changed them
        method = self.methodCombo.currentText
        if method not in SART_DEFAULTS:
            return
        current = (int(self.itersSpin.value), int(self.subsetsSpin.value), round(float(self.relaxSpin.value), 2))
        if current == SART_DEFAULTS[self.lastIterativeMethod]:
            iterations, subsets, relaxation = SART_DEFAULTS[method]
            self.itersSpin.setValue(iterations)
            self.subsetsSpin.setValue(subsets)
            self.relaxSpin.setValue(relaxation)
        self.lastIterativeMethod = method

    def _updateCutoffV(self, value):
        # Keep the vertical cut-off equal to the cut-off unless the user has set it apart
        if round(float(self.cutoffVSpin.value), 2) == round(self.lastCutoff, 2):
            self.cutoffVSpin.setValue(value)
        self.lastCutoff = float(value)

    def _updateUnitsState(self, *args):
        self.muWaterSpin.setEnabled(self.unitsCombo.currentText == UNITS_HU)
        self.muWaterSpin.setToolTip("Water attenuation for the HU scale (fixed-water mode, and for subtracted runs).")

    def _watchSettings(self):
        watched = [self.intensityCombo, self.i0Combo, self.i0Spin, self.binningCombo, self.subtractCombo,
                   self.flipColumnsCheckBox, self.flipRowsCheckBox, self.negateCheckBox, self.angleOffsetSpin,
                   self.offsetUSpin, self.offsetVSpin, self.methodCombo, self.filterCombo, self.cutoffSpin,
                   self.cutoffVSpin, self.truncationSpin, self.itersSpin, self.subsetsSpin, self.relaxSpin,
                   self.matrixCombo, self.voxelSpin, self.slicesSpin, self.unitsCombo, self.muWaterSpin,
                   self.cropCheckBox, self.calibrationCheckBox]
        for widget in watched:
            if isinstance(widget, qt.QDoubleSpinBox):
                signal = "valueChanged(double)"
            elif isinstance(widget, qt.QSpinBox):
                signal = "valueChanged(int)"
            elif isinstance(widget, qt.QComboBox):
                signal = "currentIndexChanged(int)"
            else:
                signal = "toggled(bool)"
            widget.connect(signal, self._updateLoadState)

    # ---------------- settings ----------------
    def currentParams(self):
        subtractIndex = self.subtractCombo.currentIndex
        return {
            "intensity": self.intensityCombo.currentText,
            "i0Mode": self.i0Combo.currentText,
            "i0Value": float(self.i0Spin.value),
            "binning": BINNINGS[self.binningCombo.currentIndex],
            "subtract": ((self.otherRuns[subtractIndex - 1]["file"], self.otherRuns[subtractIndex - 1].get("sweep", 0))
                         if subtractIndex > 0 else None),
            "useCalibration": bool(self.calibrationCheckBox.checked),
            "flipColumns": bool(self.flipColumnsCheckBox.checked),
            "flipRows": bool(self.flipRowsCheckBox.checked),
            "negateAngles": bool(self.negateCheckBox.checked),
            "angleOffsetDeg": float(self.angleOffsetSpin.value),
            "offsetU": float(self.offsetUSpin.value),
            "offsetV": float(self.offsetVSpin.value),
            "method": self.methodCombo.currentText,
            "filter": self.filterCombo.currentText,
            "cutoff": float(self.cutoffSpin.value),
            "cutoffV": float(self.cutoffVSpin.value),
            "truncation": int(self.truncationSpin.value) / 100.0,
            "iterations": int(self.itersSpin.value),
            "subsets": int(self.subsetsSpin.value),
            "relaxation": float(self.relaxSpin.value),
            "matrix": MATRIX_SIZES[self.matrixCombo.currentIndex],
            "voxelMm": float(self.voxelSpin.value),
            "slices": int(self.slicesSpin.value),
            "units": self.unitsCombo.currentText,
            "muWater": float(self.muWaterSpin.value),
            "cropFov": bool(self.cropCheckBox.checked),
            "forceCpu": bool(self.cpuCheckBox.checked),
        }

    def _geometryOptions(self, params):
        return {key: params[key] for key in ("flipColumns", "flipRows", "negateAngles", "angleOffsetDeg",
                                             "offsetU", "offsetV", "binning", "useCalibration")}

    # ---------------- logging / progress ----------------
    def log(self, message):
        logging.info(f"CBCT recon: {message}")
        if not hasattr(self, "logText"):
            self.pendingLog.append(message)
            return
        self.logText.appendPlainText(message)
        slicer.app.processEvents()

    def _progress(self, label, done, total, text):
        self.progressBar.setRange(0, max(1, total))
        self.progressBar.setValue(done)
        self.progressLabel.setText(text)
        slicer.app.processEvents()

    def _setBusy(self, busy):
        self.busy = busy
        if busy:
            self.abortRequested = False
            self.stopProjCine()
        elif self.cineCheckBox.checked:
            self.startProjCine()
        self.applyButton.setEnabled(not busy)
        self.applyButton.setText("Reconstructing..." if busy else "Reconstruct")
        self.cancelButton.setText("Stop" if busy else "Cancel")
        self.cancelButton.setEnabled(True)
        if not busy:
            self.progressBar.setRange(0, 1)
            self.progressBar.setValue(0)
            self.progressLabel.setText("")
        self._updateLoadState()

    def requestStop(self):
        if self.busy and not self.abortRequested:
            self.abortRequested = True
            self.cancelButton.setEnabled(False)
            self.log("Stopping...")

    def onCancelClicked(self):
        if self.busy:
            self.requestStop()
        else:
            self.dialog.reject()

    def _isUpToDate(self):
        return (self.image is not None and self.appliedParams is not None
                and self.appliedParams == self._resultParams(self.currentParams()))

    @staticmethod
    def _resultParams(params):
        """Settings that change the result (not where it is computed or the water value found)."""
        return {key: value for key, value in params.items() if key not in ("muWaterUsed", "forceCpu")}

    def _updateLoadState(self, *args):
        if not hasattr(self, "loadButton"):
            return
        upToDate = self._isUpToDate()
        self.loadButton.setEnabled(upToDate and not self.busy)
        if self.busy:
            text = ""
        elif self.lastRunFailed:
            text = ("<span style='color:#d04040'>&#9888; The last reconstruction did not finish (see the log) - "
                    "press <b>Reconstruct</b> again before loading.</span>")
        elif self.image is None:
            text = "<i>Press Reconstruct to preview the result.</i>"
        elif not upToDate:
            text = ("<span style='color:#d08000'>&#9888; Settings changed since the last reconstruction - "
                    "press <b>Reconstruct</b> to update the preview before loading.</span>")
        else:
            text = "<span style='color:#2e8b3d'>&#10004; Preview matches the settings - ready to load.</span>"
        self.staleNote.setText(text)

    # ---------------- run ----------------
    def run(self):
        """Show the window. Returns (sitkImage, info) or (None, None) when cancelled."""
        try:
            self.log(xadata.describeGeometry(self.geometry))
            self.log(f"Angles from {self.geometry['angleSource']}; distances from {self.geometry['distanceSource']}.")
            for note in self.geometry.get("notes", []):
                self.log(f"Note: {note}")
            for warning in self.geometry.get("warnings", []):
                self.log(f"WARNING: {warning}")
            if len(self.runGeometry.get("sweeps", [])) > 1:
                start, stop = self.geometry["frameRange"]
                label = xadata.sweepLabels(self.runGeometry)[self.sweepIndex]
                self.log(f"Reconstructing {label}: frames {start + 1}-{stop} of {self.runGeometry['nFrames']}.")
            if self.otherRuns:
                self.log(f"{len(self.otherRuns)} sweep(s) with the same trajectory found; "
                         "they can be subtracted under Projections > Subtract run.")
            if self.geometry.get("calibration") is not None:
                from . import gecalib
                _, calibrationInfo = gecalib.frameMatrices(self.geometry)
                if "reason" in calibrationInfo:
                    self.log(f"WARNING: GE 3D calibration not usable: {calibrationInfo['reason']}.")
                    self.calibrationCheckBox.setChecked(False)
                    self.calibrationCheckBox.setEnabled(False)
                else:
                    self.log(gecalib.describe(calibrationInfo, self.geometry["calibration"]))
            try:
                device = recon.torchDevice()
                self.log(f"Reconstructing on {'the GPU' if device.type == 'cuda' else 'the CPU'} (PyTorch).")
                if device.type == "cuda":
                    import torch
                    memoryGb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                    self.log(f"GPU: {torch.cuda.get_device_name(0)}, {memoryGb:.1f} GB. A 512 matrix needs "
                             "about 3-4 GB for SART without binning; if it is much slower than 256, tick 'Use the CPU'.")
                else:
                    self.cpuCheckBox.setChecked(True)
                    self.cpuCheckBox.setEnabled(False)
                    self.log("Note: without a GPU, FDK of a full run takes one to a few minutes; SART is much "
                             "slower - use 2 x 2 binning and a 128-192 matrix for it.")
            except Exception:
                pass
            with slicer.util.WaitCursor():
                self.allFrames = xadata.readFrames(self.inputDcm, self.geometry, log=self.log)
                start, stop = self.geometry["frameRange"]
                self.frames = self.allFrames[start:stop]
            self._showProjections()
            qt.QTimer.singleShot(0, self.onApply)
            result = self.dialog.exec_()
        finally:
            self.stopProjCine()
            self.renderer.cleanup()
        if result != qt.QDialog.Accepted or self.image is None:
            return None, None
        return self.image, self.info

    def _lineIntegrals(self, params):
        key = (params["intensity"], params["i0Mode"], params["i0Value"], params["binning"], params["subtract"])
        if self.integrals is not None and self.integralsKey == key:
            return self.integrals
        self.integrals = None
        collimator = recon.binnedCollimator(self.geometry.get("collimator"), params["binning"],
                                            self.geometry["rows"] // params["binning"],
                                            self.geometry["columns"] // params["binning"])

        def convert(frames):
            binned = recon.binFrames(frames, params["binning"])
            i0 = recon.estimateI0(binned, params["i0Mode"], params["i0Value"], collimator)
            if np.ndim(i0) == 0:
                self.log(f"I0 = {float(i0):.0f}")
            return recon.lineIntegrals(binned, params["intensity"], i0, collimator)

        integrals = convert(self.frames)
        if params["subtract"]:
            other = self.otherFrames.get(params["subtract"])
            if other is None:
                otherFile, otherSweep = params["subtract"]
                if otherFile == self.inputDcm:
                    otherGeometry = xadata.sweepGeometry(self.runGeometry, otherSweep)
                    start, stop = otherGeometry["frameRange"]
                    other = self.allFrames[start:stop]
                else:
                    otherGeometry = xadata.sweepGeometry(
                        xadata.readGeometry(xadata.readHeader(otherFile), log=self.log), otherSweep)
                    start, stop = otherGeometry["frameRange"]
                    other = xadata.readFrames(otherFile, otherGeometry, log=self.log)[start:stop]
                if other.shape != self.frames.shape:
                    raise ValueError("The sweep to subtract has a different number or size of frames.")
                difference = np.abs(otherGeometry["primaryDeg"] - self.geometry["primaryDeg"]).max()
                if difference > 2.0:
                    raise ValueError(f"The sweep to subtract follows a different trajectory "
                                     f"(angles differ by up to {difference:.1f} deg).")
                self.log(f"Subtracting {self.subtractCombo.currentText} (angles differ by at most {difference:.2f} deg).")
                self.otherFrames = {params["subtract"]: other}
            integrals = integrals - convert(other)
        self.integrals = integrals.astype(np.float32)
        self.integralsKey = key
        return self.integrals

    def _grid(self, params, projectionGeometry):
        grid = geom.defaultGrid(projectionGeometry, params["matrix"], params["voxelMm"] or None,
                                maxSlices=None)
        if params["slices"]:
            grid = geom.ReconGrid((grid.nx, grid.ny, params["slices"]), (grid.sx, grid.sy, grid.sz))
        return grid

    def onApply(self):
        if self.busy:
            return
        params = self.currentParams()
        self._setBusy(True)
        self.lastRunFailed = False
        started = time.time()
        try:
            projectionGeometry = geom.ProjectionGeometry(self.geometry, self._geometryOptions(params))
            grid = self._grid(params, projectionGeometry)
            self.log(f"Volume {grid.nx} x {grid.ny} x {grid.nz}, voxel {grid.sx:.2f} mm "
                     f"(field of view {2 * projectionGeometry.fovRadius():.0f} mm wide, "
                     f"{projectionGeometry.fovHeight():.0f} mm high)")
            p = self._lineIntegrals(params)
            volume = self._reconstruct(p, projectionGeometry, grid, params)
        except recon.ReconstructionAborted:
            self.log("Reconstruction stopped.")
            self.lastRunFailed = self.image is not None
            self._setBusy(False)
            return
        except Exception as e:
            logging.error(traceback.format_exc())
            self.log(f"ERROR: {e}")
            self.lastRunFailed = True
            self._setBusy(False)
            return

        output, params["muWaterUsed"] = self._toUnits(volume, params)
        self._setResult(output, grid, params, projectionGeometry)
        self.log(f"Done in {time.time() - started:.0f} s.")
        self._setBusy(False)

    def _toUnits(self, volume, params):
        """Output array and the water attenuation (1/cm) used for HU, or None."""
        if params["units"] not in HU_UNITS:
            return recon.toOutputUnits(volume, "1/cm"), None
        muWater = params["muWater"]
        if params["subtract"]:
            # Subtracted (DSA) volumes hold only the contrast: HU of enhancement, no water offset
            if params["units"] == UNITS_HU_AUTO:
                self.log(f"Subtracted run: the soft-tissue peak is gone, so the fixed water value "
                         f"({muWater:.3f} /cm) is used; values are the contrast enhancement in HU.")
            return (1000.0 * volume * 10.0 / muWater).astype(np.float32), muWater
        if params["units"] == UNITS_HU_AUTO:
            estimate = recon.estimateWaterAttenuation(volume)
            if estimate is None:
                self.log(f"Could not find a soft-tissue peak; using the fixed water value {muWater:.3f} /cm.")
            else:
                muWater = estimate * 10.0
                self.log(f"Water attenuation from the soft-tissue peak: {muWater:.3f} /cm "
                         "(soft tissue set to 40 HU).")
        output = recon.toOutputUnits(volume, "HU", muWater)
        output[volume == 0] = -1000.0  # outside the field of view reads as air
        return output, muWater

    def _reconstruct(self, p, projectionGeometry, grid, params):
        import torch
        device = recon.torchDevice(preferGpu=not params["forceCpu"])
        self.log(f"Reconstructing on {'the GPU' if device.type == 'cuda' else 'the CPU'}.")
        kwargs = {"progress": self._progress, "shouldAbort": lambda: self.abortRequested}
        for attempt in range(2):
            try:
                volume = None
                if params["method"] in (METHOD_FDK, METHOD_FDK_SART):
                    volume = recon.fdk(p, projectionGeometry, grid, params["filter"], params["cutoff"],
                                       params["truncation"], device=device, maskOutsideFov=params["cropFov"],
                                       log=self.log, cutoffV=params.get("cutoffV"), **kwargs)
                if params["method"] in (METHOD_FDK_SART, METHOD_SART):
                    volume = recon.sart(p, projectionGeometry, grid, params["iterations"], params["subsets"],
                                        params["relaxation"], initial=volume, device=device, log=self.log,
                                        **kwargs)
                    if params["cropFov"]:
                        volume[~recon.fovMask(projectionGeometry, grid, device=device, **kwargs)] = 0.0
                if not params["cropFov"]:
                    self.log("Not cropped: voxels outside the circle are missed by some views, so expect "
                             "streaks and wrong values there.")
                return volume
            except RuntimeError as e:  # torch.cuda.OutOfMemoryError is a RuntimeError
                if device.type != "cuda" or attempt or "out of memory" not in str(e).lower():
                    raise
                self.log("GPU out of memory; retrying on the CPU.")
                torch.cuda.empty_cache()
                device = torch.device("cpu")

    def _setResult(self, output, grid, params, projectionGeometry):
        import SimpleITK as sitk
        image = sitk.GetImageFromArray(output)
        image.SetSpacing((grid.sx, grid.sy, grid.sz))
        image.SetOrigin(tuple(float(v) for v in grid.origin))
        image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # LPS, isocenter at 0
        self.image = image
        self.reconArray = output
        self.reconGrid = grid
        hounsfield = params["units"] in HU_UNITS and not params["subtract"]
        self.reconWindow = display.displayWindow(output[output > -1000.0] if hounsfield else output)
        self.appliedParams = self._resultParams(params)
        self.info = self._info(params, grid, projectionGeometry)
        for slider, size in ((self.axialSlider, grid.nz), (self.coronalSlider, grid.ny),
                             (self.sagittalSlider, grid.nx)):
            slider.blockSignals(True)
            slider.setRange(0, size - 1)
            if slider.value == 0 or slider.value >= size:
                slider.setValue(size // 2)
            slider.setEnabled(True)
            slider.blockSignals(False)
        self.updateReconViews()

    def _info(self, params, grid, projectionGeometry):
        info = {
            "Method": params["method"],
            "Units": ("HU" if params["units"] in HU_UNITS else "1/cm"),
            "Geometry": ("GE 3D calibration" if projectionGeometry.calibrated else "nominal angles"),
            "Intensity": params["intensity"],
            "I0": params["i0Mode"] + (f" {params['i0Value']:.0f}" if params["i0Mode"] == recon.I0_MANUAL else ""),
            "Binning": params["binning"],
            "Truncation": f"{params['truncation']:.2f}",
            "GeometryCorrections": (f"mirrorColumns={params['flipColumns']} mirrorRows={params['flipRows']} "
                                    f"negateAngles={params['negateAngles']} angleOffset={params['angleOffsetDeg']} "
                                    f"shiftU={params['offsetU']} shiftV={params['offsetV']}"),
            "ScanRangeDeg": f"{math.degrees(projectionGeometry.scanRange):.1f}",
            "Views": projectionGeometry.nViews,
            "VoxelMm": f"{grid.sx:.3f}",
        }
        if params["method"] != METHOD_SART:
            info["Filter"] = f"{params['filter']} cutoff {params['cutoff']:.2f}, vertical Hann {params['cutoffV']:.2f}"
        if params["method"] != METHOD_FDK:
            info["SART"] = f"{params['iterations']} iterations x {params['subsets']} subsets, relaxation {params['relaxation']}"
        if params.get("muWaterUsed"):
            info["MuWaterPerCm"] = f"{params['muWaterUsed']:.4f}"
        if len(self.runGeometry.get("sweeps", [])) > 1:
            info["Sweep"] = xadata.sweepLabels(self.runGeometry)[self.sweepIndex]
            info["Frames"] = "%d-%d" % (self.geometry["frameRange"][0] + 1, self.geometry["frameRange"][1])
        if params["subtract"]:
            info["SubtractedRun"] = self.subtractCombo.currentText
        # The preview's grey scale, so the loaded volume is shown the same way
        info["DisplayWindow"] = "%.1f %.1f" % self.reconWindow
        try:
            import torch
            info["PyTorch"] = torch.__version__
        except Exception:
            pass
        return info

    def onLoad(self):
        if not self._isUpToDate():
            self._updateLoadState()
            return
        self.dialog.accept()

    # ---------------- views ----------------
    def _showProjections(self):
        if self.frames is None:
            return
        self.projWindow = display.displayWindow(self.frames[:: max(1, self.frames.shape[0] // 20)], 0.5, 99.5)
        self.projSlider.setRange(0, self.frames.shape[0] - 1)
        self.projSlider.setEnabled(True)
        self.projSlider.setValue(self.frames.shape[0] // 2)
        self.updateProjView(self.projSlider.value)

    def updateProjView(self, index):
        if self.frames is None:
            return
        index = max(0, min(self.frames.shape[0] - 1, int(index)))
        rgb = display.grayscale(self.frames[index], self.projWindow)
        self.projView.setPixmap(self.renderer.toPixmap(rgb))
        self.projInfo.setText(f"frame {index + 1}/{self.frames.shape[0]}, "
                              f"primary {self.geometry['primaryDeg'][index]:.1f} deg")

    def updateReconViews(self):
        if self.reconArray is None:
            return
        volume = self.reconArray
        grid = self.reconGrid
        z = self.axialSlider.value
        y = self.coronalSlider.value
        x = self.sagittalSlider.value
        # Radiological display: patient's right on the left of the screen, anterior up, head up.
        # Array axes are LPS (x to the left, y posterior, z superior).
        axial = volume[z]                         # [y, x]: anterior row 0, right column 0
        coronal = volume[::-1, y, :]              # [z, x]: head up
        sagittal = volume[::-1, :, x]             # [z, y]: head up, anterior left
        self.axialView.setPixmap(self._slicePixmap(axial, grid.sy / grid.sx))
        self.coronalView.setPixmap(self._slicePixmap(coronal, grid.sz / grid.sx))
        self.sagittalView.setPixmap(self._slicePixmap(sagittal, grid.sz / grid.sy))
        zMm = grid.origin[2] + z * grid.sz
        self.axialInfo.setText(f"slice {z + 1}/{grid.nz} (S {zMm:.1f} mm)")
        self.coronalInfo.setText(f"slice {y + 1}/{grid.ny}")
        self.sagittalInfo.setText(f"slice {x + 1}/{grid.nx}")

    def _slicePixmap(self, array, aspect):
        rgb = display.grayscale(np.ascontiguousarray(array), self.reconWindow)
        if abs(aspect - 1.0) > 0.01:
            rows = max(1, int(round(rgb.shape[0] * aspect)))
            index = np.clip((np.arange(rows) / aspect).astype(int), 0, rgb.shape[0] - 1)
            rgb = rgb[index]
        return self.renderer.toPixmap(rgb)

    # ---------------- cine ----------------
    def onCineToggled(self, checked):
        self.startProjCine() if checked else self.stopProjCine()

    def onFpsChanged(self, fps):
        if self.projCineTimer is not None:
            self.projCineTimer.setInterval(int(1000 / max(1, fps)))

    def startProjCine(self):
        if self.frames is None or self.busy:
            return
        if self.projCineTimer is None:
            self.projCineTimer = qt.QTimer()
            self.projCineTimer.connect("timeout()", self.advanceProjCine)
        self.projCineTimer.setInterval(int(1000 / max(1, self.fpsSpin.value)))
        self.projCineTimer.start()

    def stopProjCine(self):
        if self.projCineTimer is not None:
            self.projCineTimer.stop()

    def advanceProjCine(self):
        if self.frames is None:
            return
        self.projSlider.setValue((self.projSlider.value + 1) % self.frames.shape[0])

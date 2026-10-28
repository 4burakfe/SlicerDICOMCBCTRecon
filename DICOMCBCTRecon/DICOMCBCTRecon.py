"""
DICOM Cone-Beam CT Reconstruction for 3D Slicer

Recognizes rotational X-ray angiography runs (C-arm "spin" acquisitions: one multi-frame XA/XRF
object whose positioner angle changes from frame to frame over at least 120 degrees) in the DICOM
browser. On Load, opens a reconstruction window with the raw projections in cine, axial/coronal/
sagittal previews and the reconstruction settings:
    FDK            filtered backprojection for circular cone-beam data with Parker short-scan
                   weighting (displaced-detector weighting for an offset detector over a full rotation),
                   cosine pre-weighting, apodised ramp filter, vertical Hann window, lateral truncation
                   correction
    FDK + SART     FDK followed by a few ordered-subsets SART iterations
    SART           iterative only (start from zero)
Geometry comes from the DICOM header: per-frame primary/secondary angles (PositionerPrimaryAngle
+ PositionerPrimaryAngleIncrement, or the enhanced XA per-frame sequences), source-to-detector and
source-to-isocenter distances (per frame from GE's private GEMS_DL_IMG_01 block when present) and
the detector pixel size (ImagerPixelSpacing). Mirroring, angle offset and detector shift can be
corrected in the window when a vendor stores the images differently.

When the study contains another run with the same trajectory (for example a contrast-filled and a
native spin), it can be subtracted in the projection domain (3D DSA).

Reconstructions are in 1/cm (attenuation) or in a Hounsfield-like scale with a chosen water value;
display-processed frames (PixelIntensityRelationship DISP) give relative, not calibrated, values.

PyTorch comes from Slicer's PyTorch extension (GPU when available). The reconstruction code is in
the DICOMCBCTReconLib package (see its __init__.py).
"""

import logging
import os
import sys
import traceback

import qt
import slicer
from DICOMLib import DICOMLoadable, DICOMPlugin
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleTest)

# The helper package lives next to this file; make it importable through "Additional module paths" too
_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

# (file, sweep) -> other sweeps of the study that follow the same trajectory, filled in by examine()
_OTHER_RUNS_FOR_SWEEP = {}
# (file, loadable name) -> sweep index (Qt-wrapped loadables can lose Python attributes)
_SWEEP_FOR_LOADABLE = {}

ROTATIONAL_MODALITIES = ("XA", "XRF", "RF")
MIN_FRAMES = 30


# ---------------------------------------------------------------------------
# Module registration (hidden: used through the DICOM browser)
# ---------------------------------------------------------------------------
class DICOMCBCTRecon(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "DICOM Cone-Beam CT Reconstruction"
        parent.categories = ["Developer Tools.DICOM Plugins"]
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Loads rotational C-arm X-ray runs (XA/XRF multi-frame) from DICOM and reconstructs them "
            "with FDK (filtered backprojection with Parker short-scan weighting) or SART, with an "
            "interactive preview before loading.")
        parent.acknowledgementText = "Reconstruction implemented with PyTorch."
        parent.hidden = True

        try:
            slicer.modules.dicomPlugins
        except AttributeError:
            slicer.modules.dicomPlugins = {}
        slicer.modules.dicomPlugins["DICOMCBCTRecon"] = DICOMCBCTReconPluginClass


class DICOMCBCTReconLogic(ScriptedLoadableModuleLogic):
    """Reconstruction without the window, for scripts:
        logic = DICOMCBCTReconLogic()
        image, info = logic.reconstructFile(path)   # SimpleITK image (1/cm), settings"""

    def reconstructFile(self, path, matrix=256, binning=1, filterName=None, truncation=0.25,
                        geometryOptions=None, sweep=0, log=logging.info):
        from DICOMCBCTReconLib import geometry as geom
        from DICOMCBCTReconLib import reconstruction as recon
        from DICOMCBCTReconLib import xadata
        import SimpleITK as sitk

        geometry = xadata.sweepGeometry(xadata.readGeometry(xadata.readHeader(path), log=log), sweep)
        start, stop = geometry["frameRange"]
        frames = recon.binFrames(xadata.readFrames(path, geometry, log=log)[start:stop], binning)
        options = dict(geometryOptions or {}, binning=binning)
        projectionGeometry = geom.ProjectionGeometry(geometry, options)
        collimator = recon.binnedCollimator(geometry.get("collimator"), binning,
                                            projectionGeometry.rows, projectionGeometry.columns)
        p = recon.lineIntegrals(frames, recon.defaultIntensityMode(geometry), recon.estimateI0(frames, recon.I0_GLOBAL,
                                                                              collimator=collimator), collimator)
        grid = geom.defaultGrid(projectionGeometry, matrix)
        volume = recon.fdk(p, projectionGeometry, grid, filterName or recon.DEFAULT_FILTER, 1.0, truncation, log=log)
        image = sitk.GetImageFromArray(recon.toOutputUnits(volume))
        image.SetSpacing((grid.sx, grid.sy, grid.sz))
        image.SetOrigin(tuple(float(v) for v in grid.origin))
        return image, {"Method": "FDK", "Units": "1/cm", "Matrix": matrix, "Binning": binning}


# ---------------------------------------------------------------------------
# DICOM plugin
# ---------------------------------------------------------------------------
class DICOMCBCTReconPluginClass(DICOMPlugin):
    def __init__(self):
        super().__init__()
        self.loadType = "Cone-beam CT (rotational X-ray)"
        self.tags["modality"] = "0008,0060"
        self.tags["numberOfFrames"] = "0028,0008"
        self.tags["seriesDescription"] = "0008,103e"
        self.tags["instanceNumber"] = "0020,0013"
        self.tags["acquisitionTime"] = "0008,0032"
        self.tags["studyInstanceUID"] = "0020,000d"

    def examineForImport(self, fileLists):
        return self.examine(fileLists)

    def examine(self, fileLists):
        loadables = []
        for files in fileLists:
            loadables += self.examineFiles(files)
        return loadables

    def _candidateFile(self, filePath):
        """Cheap database check before the header is parsed."""
        db = slicer.dicomDatabase
        modality = (db.fileValue(filePath, self.tags["modality"]) or "").strip().upper()
        if modality not in ROTATIONAL_MODALITIES:
            return False
        try:
            return int(float(db.fileValue(filePath, self.tags["numberOfFrames"]) or 1)) >= MIN_FRAMES
        except ValueError:
            return False

    def _runSummary(self, filePath):
        """Geometry summary of a rotational run, or None when the file is not one."""
        from DICOMCBCTReconLib import xadata
        try:
            header = xadata.readHeader(filePath)
            ok, reason = xadata.isRotationalRun(header)
            if not ok:
                logging.debug(f"CBCT recon: {os.path.basename(filePath)} skipped: {reason}")
                return None
            geometry = xadata.readGeometry(header, log=lambda *args: None)
            return geometry
        except Exception:
            logging.debug("CBCT recon: could not read " + filePath + "\n" + traceback.format_exc())
            return None

    def _runLabel(self, filePath):
        db = slicer.dicomDatabase
        number = (db.fileValue(filePath, self.tags["instanceNumber"]) or "").strip()
        time = (db.fileValue(filePath, self.tags["acquisitionTime"]) or "").strip()
        clock = f"{time[0:2]}:{time[2:4]}:{time[4:6]}" if len(time) >= 6 else ""
        return " ".join(part for part in (f"run #{number}" if number else "run", f"acquired {clock}" if clock else "") if part)

    def _sweepName(self, filePath, geometry, sweep):
        from DICOMCBCTReconLib import xadata
        label = xadata.sweepLabels(geometry)[sweep]
        return " ".join(part for part in (self._runLabel(filePath), label) if part)

    def _matchingRuns(self, filePath, geometry, sweep, summaries=None):
        """Other sweeps of this run and of the study's other runs whose trajectory matches this
        sweep (candidates for subtraction)."""
        import numpy as np
        from DICOMCBCTReconLib import xadata
        db = slicer.dicomDatabase
        target = xadata.sweepGeometry(geometry, sweep)
        matches = []
        try:
            files = [filePath]
            studyUID = db.fileValue(filePath, self.tags["studyInstanceUID"]) or ""
            if studyUID:
                for seriesUID in db.seriesForStudy(studyUID):
                    files += [f for f in db.filesForSeries(seriesUID) if f != filePath]
            for other in files:
                if summaries is not None and other in summaries:
                    otherGeometry = summaries[other]
                elif other == filePath:
                    otherGeometry = geometry
                else:
                    otherGeometry = self._runSummary(other) if self._candidateFile(other) else None
                if summaries is not None:
                    summaries[other] = otherGeometry
                if otherGeometry is None:
                    continue
                for k in range(len(otherGeometry["sweeps"])):
                    if other == filePath and k == sweep:
                        continue
                    candidate = xadata.sweepGeometry(otherGeometry, k)
                    if (candidate["nFrames"] != target["nFrames"] or candidate["rows"] != target["rows"]
                            or candidate["columns"] != target["columns"]):
                        continue
                    if np.abs(candidate["primaryDeg"] - target["primaryDeg"]).max() > 2.0:
                        continue
                    matches.append({"file": other, "sweep": k,
                                    "label": self._sweepName(other, otherGeometry, k)})
        except Exception:
            logging.warning("CBCT recon: could not look for matching runs:\n" + traceback.format_exc())
        return matches

    def examineFiles(self, files):
        from DICOMCBCTReconLib import xadata
        loadables = []
        summaries = {}
        for filePath in files:
            if not self._candidateFile(filePath):
                continue
            geometry = summaries.get(filePath) or self._runSummary(filePath)
            summaries[filePath] = geometry
            if geometry is None:
                continue
            desc = slicer.dicomDatabase.fileValue(filePath, self.tags["seriesDescription"]) or "Rotational run"
            nSweeps = len(geometry["sweeps"])
            for sweep in range(nSweeps):
                sweepGeometry = xadata.sweepGeometry(geometry, sweep)
                if sweepGeometry["arcDeg"] < xadata.MIN_ARC_DEG or sweepGeometry["nFrames"] < MIN_FRAMES:
                    continue
                _OTHER_RUNS_FOR_SWEEP[(filePath, sweep)] = self._matchingRuns(filePath, geometry, sweep, summaries)
                loadable = DICOMLoadable()
                loadable.files = [filePath]
                loadable.name = f"{desc} {self._sweepName(filePath, geometry, sweep)} CBCT"
                start, stop = sweepGeometry["frameRange"]
                loadable.tooltip = (f"Rotational X-ray run, {sweepGeometry['nFrames']} frames over "
                                    f"{sweepGeometry['arcDeg']:.0f} deg"
                                    + (f" (frames {start + 1}-{stop} of {geometry['nFrames']})" if nSweeps > 1 else "")
                                    + (", GE 3D calibration available" if geometry.get("calibration") is not None else "")
                                    + " - reconstructed as cone-beam CT on load")
                # In a subtraction spin, offer the contrast sweep first; the mask is mostly for subtraction
                loadable.selected = not (nSweeps == 2 and geometry.get("subtractionSpin") and sweep == 0)
                loadable.confidence = 0.95  # above the generic scalar volume / multi-frame plugins
                _SWEEP_FOR_LOADABLE[(filePath, loadable.name)] = sweep
                loadables.append(loadable)
        return loadables

    def load(self, loadable):
        if not ensureDependencies():
            return None
        _reloadLibInDeveloperMode()
        from DICOMCBCTReconLib.dialog import CbctReconDialog

        filePath = loadable.files[0]
        sweep = _SWEEP_FOR_LOADABLE.get((filePath, loadable.name), 0)
        otherRuns = _OTHER_RUNS_FOR_SWEEP.get((filePath, sweep))
        if otherRuns is None:
            geometry = self._runSummary(filePath)
            otherRuns = self._matchingRuns(filePath, geometry, sweep) if geometry else []
        try:
            dialog = CbctReconDialog(filePath, loadable.name, otherRuns, sweepIndex=sweep)
        except Exception as e:
            slicer.util.errorDisplay(f"Cannot reconstruct this run: {e}", detailedText=traceback.format_exc())
            return None
        image, info = dialog.run()
        if image is None:
            logging.info("Cone-beam reconstruction cancelled by user.")
            return None

        import sitkUtils
        method = "FDK" if info.get("Method", "").startswith("FDK") else "SART"
        if info.get("SART") and method == "FDK":
            method = "FDK+SART"
        suffix = f"{method}" + (" DSA" if info.get("SubtractedRun") else "")
        nodeName = slicer.mrmlScene.GenerateUniqueName(f"{loadable.name} {suffix}")
        volumeNode = sitkUtils.PushVolumeToSlicer(image, name=nodeName)
        for key, value in info.items():
            volumeNode.SetAttribute(f"CBCTRecon.{key}", str(value))
        try:
            self.addSeriesInSubjectHierarchy(loadable, volumeNode)
        except Exception:
            logging.warning("Could not place volume in subject hierarchy:\n" + traceback.format_exc())

        volumeNodeID = volumeNode.GetID()
        hounsfield = info.get("Units") == "HU" and not info.get("SubtractedRun")
        try:
            window = tuple(float(v) for v in info.get("DisplayWindow", "").split())
        except ValueError:
            window = ()
        window = window if len(window) == 2 else None
        qt.QTimer.singleShot(0, lambda: showAndCenterVolume(volumeNodeID, hounsfield, window))
        return volumeNode


def showAndCenterVolume(volumeNodeID, hounsfield=False, window=None):
    """Show the new volume in all slice views and center on it.

    window = (minimum, maximum) of the preview's grey scale, so the volume looks the same as in the
    reconstruction window; without it HU volumes get a CT-like window and others an automatic one.
    """
    try:
        volumeNode = slicer.mrmlScene.GetNodeByID(volumeNodeID)
        if not volumeNode:
            return
        volumeNode.CreateDefaultDisplayNodes()
        displayNode = volumeNode.GetDisplayNode()
        if displayNode:
            displayNode.SetInterpolate(True)
            if window:
                displayNode.SetAutoWindowLevel(False)
                displayNode.SetWindowLevelMinMax(window[0], window[1])
            elif hounsfield:
                displayNode.SetAutoWindowLevel(False)
                displayNode.SetWindowLevel(1500, 300)
            else:
                displayNode.SetAutoWindowLevel(True)
        slicer.util.setSliceViewerLayers(background=volumeNode, foreground=None, fit=True)
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
        slicer.modules.markups.logic().JumpSlicesToLocation(center[0], center[1], center[2], True)
    except Exception:
        logging.warning("Could not center views on the reconstructed volume:\n" + traceback.format_exc())


def _reloadLibInDeveloperMode():
    """With Slicer's developer mode on, pick up edits to the helper package without restarting."""
    try:
        if not slicer.util.settingsValue("Developer/DeveloperMode", False, converter=slicer.util.toBool):
            return
        import importlib
        import DICOMCBCTReconLib.xadata as xadata
        import DICOMCBCTReconLib.gecalib as gecalib
        import DICOMCBCTReconLib.geometry as geometry
        import DICOMCBCTReconLib.reconstruction as reconstruction
        import DICOMCBCTReconLib.display as display
        import DICOMCBCTReconLib.dialog as dialog
        for module in (xadata, gecalib, geometry, reconstruction, display, dialog):  # dependency order
            importlib.reload(module)
    except Exception:
        logging.warning("Could not reload DICOMCBCTReconLib:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Dependencies (same policy as the SPECT plugin: torch only through the PyTorch extension)
# ---------------------------------------------------------------------------
def ensureDependencies():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import PyTorchUtils
    except ImportError:
        message = ("Cone-beam reconstruction needs PyTorch, which is provided by Slicer's 'PyTorch' extension.\n\n"
                   "Install the 'PyTorch' extension from the Extensions Manager, restart Slicer, "
                   "then load the run again.")
        if hasattr(slicer.app, "openExtensionsManagerDialog"):
            if slicer.util.confirmOkCancelDisplay(message + "\n\nOpen the Extensions Manager now?"):
                slicer.app.openExtensionsManagerDialog()
        else:
            slicer.util.infoDisplay(message)
        return False
    if not slicer.util.confirmOkCancelDisplay(
            "Cone-beam reconstruction needs PyTorch, which is not installed yet.\n\n"
            "Install it now with the PyTorch extension? It selects the CPU or CUDA build for "
            "this computer and can take several minutes."):
        return False
    try:
        with slicer.util.WaitCursor():
            torch = PyTorchUtils.PyTorchUtilsLogic().installTorch(askConfirmation=False)
    except Exception as e:
        slicer.util.errorDisplay(f"PyTorch installation failed: {e}", detailedText=traceback.format_exc())
        return False
    if torch is None:
        slicer.util.errorDisplay("PyTorch installation failed. You can install it from the PyTorch module "
                                 "(Utilities > PyTorch) and then load the run again.")
        return False
    return True


# ---------------------------------------------------------------------------
# Self test (synthetic phantom, no patient data)
# ---------------------------------------------------------------------------
class DICOMCBCTReconTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_SyntheticRun()

    def test_SyntheticRun(self):
        """Write a synthetic rotational run, read it back and reconstruct it with FDK."""
        import tempfile
        import numpy as np
        try:
            import torch  # noqa: F401
        except ImportError:
            self.delayDisplay("PyTorch not installed - test skipped")
            return
        from DICOMCBCTReconLib import geometry as geom
        from DICOMCBCTReconLib import phantom

        self.delayDisplay("Creating a synthetic rotational run")
        header = phantom.syntheticRunHeader(nFrames=121, arcDeg=200.0, rows=96, columns=96, pixelMm=3.2)
        lineIntegrals = phantom.projectPhantom(geom.ProjectionGeometry(header))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "synthetic_run.dcm")
            phantom.writeSyntheticRun(path, header, lineIntegrals)
            image, info = DICOMCBCTReconLogic().reconstructFile(path, matrix=96, truncation=0.0,
                                                                 log=lambda *args: None)
        import SimpleITK as sitk
        array = sitk.GetArrayFromImage(image)
        center = array[array.shape[0] // 2, array.shape[1] // 2, array.shape[2] // 2]
        # Body attenuation is 0.020 /mm = 0.20 /cm
        self.assertAlmostEqual(float(center), 0.20, delta=0.02)
        self.assertTrue(np.isfinite(array).all())
        self.delayDisplay("Synthetic run reconstructed correctly")

"""Rotational X-ray angiography (C-arm cone-beam CT) DICOM reading: acquisition geometry and frames.

No Slicer dependency (pydicom and numpy only; SimpleITK as a fallback pixel reader).

Supported input: one multi-frame XA / XRF object holding a rotational run, either
  - classic XA: PositionerPrimaryAngle + PositionerPrimaryAngleIncrement (one value per frame),
    DistanceSourceToDetector / DistanceSourceToPatient, ImagerPixelSpacing, or
  - enhanced XA: per-frame Positioner Position Sequence and X-Ray Geometry Sequence.
GE Innova runs also carry per-frame source-to-detector / source-to-isocenter distances in the
private block GEMS_DL_IMG_01 (0019,xxBE / 0019,xxE9); these are used when present.

Angles follow DICOM (patient based): primary angle positive towards LAO (detector moves to the
patient's left), secondary angle positive towards CRA (detector tilts towards the head).
"""

import logging
import struct

import numpy as np

# Rotational runs shorter than this cannot be reconstructed in any useful way
MIN_ARC_DEG = 120.0
MIN_FRAMES = 30

GE_DL_IMG_CREATOR = "GEMS_DL_IMG_01"
GE_SID_OFFSET = 0xBE   # per-frame source to detector distance (mm), FL
GE_SOD_OFFSET = 0xE9   # per-frame source to isocenter distance (mm), FL

# Tags with no keyword lookup needed elsewhere
TAG_PER_FRAME_GROUPS = 0x52009230
TAG_SHARED_GROUPS = 0x52009229


# ---------------------------------------------------------------------------
# Small helpers (same conventions as the SPECT plugin's nmdata module)
# ---------------------------------------------------------------------------
def _text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        value = "\\".join(str(v) for v in value)
    if isinstance(value, bytes):
        value = value.decode("latin1", "ignore")
    return str(value).strip().rstrip("\x00").strip()


def _floats(value):
    """DS/FL/IS value(s) -> list of floats ([] when absent or unreadable)."""
    if value is None:
        return []
    if isinstance(value, bytes):
        text = value.decode("latin1", "ignore").strip("\x00 ")
        items = text.split("\\")
    elif isinstance(value, str):
        items = value.strip().split("\\")
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    out = []
    for item in items:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _float32Array(value, expected):
    """GE per-frame FL arrays; in implicit-VR files they arrive as raw little-endian bytes."""
    if isinstance(value, bytes):
        if len(value) == 4 * expected:
            return list(struct.unpack("<%df" % expected, value))
        # Some exports store them as DS text
        values = _floats(value)
        return values if len(values) == expected else None
    values = _floats(value)
    return values if len(values) == expected else None


def _privateElement(ds, group, creator, offset):
    """Value of a private element identified by its creator string and its offset in the
    creator's block; None when absent."""
    try:
        import pydicom
        for blockElement in range(0x10, 0x100):
            tag = pydicom.tag.Tag(group, blockElement)
            if tag not in ds:
                continue
            if _text(ds[tag].value) == creator:
                target = pydicom.tag.Tag(group, (blockElement << 8) | offset)
                return ds[target].value if target in ds else None
    except Exception:
        return None
    return None


def _find(ds, keyword, depth=3):
    """First value of an attribute anywhere in ds or its nested sequences (breadth first,
    limited depth); used for enhanced objects whose attributes sit in functional groups."""
    if ds is None:
        return None
    if keyword in ds:
        return ds.get(keyword)
    if depth <= 0:
        return None
    for element in ds:
        if element.VR == "SQ" and element.value:
            for item in element.value[:1]:
                found = _find(item, keyword, depth - 1)
                if found is not None:
                    return found
    return None


def _perFrameValues(ds, keyword, nFrames):
    """Values of an attribute in every item of the Per-frame Functional Groups Sequence."""
    groups = ds.get(TAG_PER_FRAME_GROUPS)
    if groups is None or len(groups.value) != nFrames:
        return None
    values = []
    for item in groups.value:
        value = _find(item, keyword, depth=2)
        numbers = _floats(value)
        if not numbers:
            return None
        values.append(numbers[0])
    return values


GE_3DCAL_CREATOR = "GEMS_XR3DCAL_01"


def _numbers(value, expected=None):
    """Numeric private values: text (DS/IS), or raw bytes as float64 / float32 when implicit VR."""
    numbers = _floats(value)
    if numbers and (expected is None or len(numbers) == expected):
        return numbers
    if isinstance(value, bytes):
        for code, size in (("d", 8), ("f", 4)):
            if expected and len(value) == size * expected:
                return list(struct.unpack("<%d%s" % (expected, code), value))
    return numbers if expected is None else []


def readGeCalibration(ds):
    """GE Innova 3D calibration (private block GEMS_XR3DCAL_01, group 0021), when present.

    It holds one calibrated 3x4 projection matrix per calibration position (gantry angle), measured
    on a phantom, mapping 3D points (calibration frame, 0.1 mm units) to pixels of a calibration
    detector grid. Returns None when absent, otherwise a dict:
        angles [n] (deg), matrices [n, 3, 4], gridSize (rows, cols), pixelMm (row, col), fovMm,
        date, kind, meanError (mm, from the calibration report) or None."""
    def element(offset):
        return _privateElement(ds, 0x0021, GE_3DCAL_CREATOR, offset)

    sequence = element(0x0B)
    angles = _numbers(element(0x09))
    if not sequence or not angles:
        return None
    matrices = []
    for item in sequence:
        values = _numbers(_privateElement(item, 0x0021, GE_3DCAL_CREATOR, 0x0C), 12)
        if len(values) != 12:
            raise ValueError("projection matrix with an unexpected number of values")
        matrices.append(np.asarray(values, dtype=float).reshape(3, 4))
    if len(matrices) != len(angles):
        raise ValueError(f"{len(matrices)} matrices for {len(angles)} calibration angles")
    rows = (_numbers(element(0x01)) or [0])[0]
    cols = (_numbers(element(0x02)) or [0])[0]
    pixel = _numbers(element(0x15))
    fov = (_numbers(element(0x03)) or [0])[0]
    if not (rows and cols and len(pixel) >= 2):
        raise ValueError("calibration detector size or pixel spacing missing")
    report = _text(element(0x20))
    meanError = None
    for line in report.splitlines():
        if line.strip().lower().startswith("gantry mean error"):
            try:
                meanError = float(line.split(":")[1])
            except (IndexError, ValueError):
                pass
    order = np.argsort(angles)
    return {
        "angles": np.asarray(angles, dtype=float)[order],
        "matrices": np.asarray(matrices)[order],
        "gridSize": (int(rows), int(cols)),
        "pixelMm": (float(pixel[0]), float(pixel[1])),
        "fovMm": float(fov),
        "date": _text(element(0x04)),
        "kind": _text(element(0x16)),
        "meanError": meanError,
    }


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------
def anglesFromIncrements(start, increments, log=logging.info, name="primary"):
    """Per-frame angles from the start angle and PositionerXAngleIncrement.

    DICOM calls the increment the change of the angle "for each frame". Vendors store it two ways:
    as the angle relative to the first frame (GE, Siemens: 0, 0.4, 0.9 ... 193) or as the step from
    the previous frame (0.5, 0.5, ...). Cumulative values start at 0 and grow steadily; steps stay
    small. Returns (angles, interpretation)."""
    increments = np.asarray(increments, dtype=float)
    if increments.size == 0:
        return np.array([start], dtype=float), "none"
    looksCumulative = abs(increments[0]) < 1e-6 and np.ptp(increments) > 5 * (np.abs(np.diff(increments)).max() + 1e-9)
    if looksCumulative:
        return start + increments, "relative to the first frame"
    steps = np.abs(increments)
    if steps.max() < 10 and np.abs(increments.sum()) > 5 * steps.max():
        log(f"Positioner {name} angle increments look like per-frame steps; summing them.")
        return start + np.concatenate([[0.0], np.cumsum(increments[1:])]), "per-frame steps"
    return start + increments, "relative to the first frame"


def findSweeps(primaryIncrements, frameTimes=None, minDropDeg=10.0):
    """Frame ranges [(start, stop), ...] of the sweeps stored in one run.

    GE "SUB SPIN" runs (3D DSA) hold a mask sweep and a contrast sweep in one object: the angle
    increment and the frame time vector restart at the first frame of the second sweep."""
    increments = np.asarray(primaryIncrements, dtype=float)
    n = increments.size
    boundaries = [0]
    for i in range(1, n):
        drop = abs(increments[i] - increments[i - 1]) > minDropDeg
        restart = frameTimes is not None and len(frameTimes) == n and frameTimes[i] == 0 and i - boundaries[-1] > 10
        if drop or restart:
            boundaries.append(i)
    boundaries.append(n)
    return [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)
            if boundaries[k + 1] - boundaries[k] > 1]


PER_FRAME_KEYS = ("primaryDeg", "secondaryDeg", "sid", "sod")


def sweepGeometry(geometry, index):
    """Geometry of one sweep: per-frame arrays sliced to its frames ('frameRange' says which)."""
    sweeps = geometry.get("sweeps") or [(0, geometry["nFrames"])]
    start, stop = sweeps[index]
    out = dict(geometry)
    for key in PER_FRAME_KEYS:
        out[key] = np.asarray(geometry[key])[start:stop]
    out["nFrames"] = stop - start
    out["frameRange"] = (start, stop)
    out["sweepIndex"] = index
    out["arcDeg"] = float(abs(out["primaryDeg"][-1] - out["primaryDeg"][0]))
    if geometry.get("calibration") is not None:
        out["calibration"] = geometry["calibration"]
    return out


def sweepLabels(geometry):
    """Short names for the sweeps of a run ('mask' / 'contrast' for two-sweep subtraction spins)."""
    sweeps = geometry.get("sweeps") or [(0, geometry["nFrames"])]
    if len(sweeps) == 1:
        return [""]
    if len(sweeps) == 2 and geometry.get("subtractionSpin"):
        return ["sweep 1 (mask)", "sweep 2 (contrast)"]
    return [f"sweep {k + 1}" for k in range(len(sweeps))]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def readHeader(path):
    import pydicom
    return pydicom.dcmread(path, stop_before_pixels=True)


def isRotationalRun(ds):
    """(True, '') when the object looks like a reconstructable rotational acquisition,
    otherwise (False, reason)."""
    modality = _text(ds.get("Modality")).upper()
    if modality not in ("XA", "XRF", "RF"):
        return False, f"modality {modality or '?'} is not XA/XRF"
    nFrames = int(_floats(ds.get("NumberOfFrames"))[0]) if _floats(ds.get("NumberOfFrames")) else 1
    if nFrames < MIN_FRAMES:
        return False, f"only {nFrames} frames"
    try:
        geometry = readGeometry(ds, log=lambda *_: None)
    except ValueError as e:
        return False, str(e)
    arcs = [sweepGeometry(geometry, k)["arcDeg"] for k in range(len(geometry["sweeps"]))]
    if max(arcs) < MIN_ARC_DEG:
        return False, f"angular range {max(arcs):.0f} deg is too short"
    return True, ""


def readGeometry(ds, log=logging.info):
    """Acquisition geometry of a rotational run. Raises ValueError when essential data is missing.

    Returns a dict with per-frame arrays (degrees, mm) and detector description:
        nFrames, rows, columns, pixelSpacing (row spacing, column spacing at the detector, mm),
        primaryDeg, secondaryDeg, sid, sod, arcDeg, angleSource, distanceSource,
        collimator (first/last valid row and column, 0-based, inclusive) or None,
        fovRotation (deg), fovFlip (bool), intensityRelationship, photometric, bitsStored,
        kvp, manufacturer, model, patientPosition, warnings, notes"""
    warnings, notes = [], []
    nFramesValues = _floats(ds.get("NumberOfFrames"))
    nFrames = int(nFramesValues[0]) if nFramesValues else 1
    rows = int(ds.get("Rows", 0) or 0)
    columns = int(ds.get("Columns", 0) or 0)
    if nFrames < 2 or rows < 2 or columns < 2:
        raise ValueError("not a multi-frame image")

    # --- angles ---
    primary = secondary = None
    angleSource = ""
    startPrimary = _floats(ds.get("PositionerPrimaryAngle"))
    startSecondary = _floats(ds.get("PositionerSecondaryAngle"))
    incPrimary = _floats(ds.get("PositionerPrimaryAngleIncrement"))
    incSecondary = _floats(ds.get("PositionerSecondaryAngleIncrement"))
    if startPrimary and len(incPrimary) == nFrames:
        primary, how = anglesFromIncrements(startPrimary[0], incPrimary, log, "primary")
        angleSource = f"PositionerPrimaryAngleIncrement ({how})"
        if len(incSecondary) == nFrames:
            secondary, _ = anglesFromIncrements(startSecondary[0] if startSecondary else 0.0,
                                                incSecondary, log, "secondary")
        else:
            secondary = np.full(nFrames, startSecondary[0] if startSecondary else 0.0)
    else:
        # Enhanced XA: one Positioner Position Sequence per frame
        perFramePrimary = _perFrameValues(ds, "PositionerPrimaryAngle", nFrames)
        if perFramePrimary is not None:
            primary = np.asarray(perFramePrimary, dtype=float)
            perFrameSecondary = _perFrameValues(ds, "PositionerSecondaryAngle", nFrames)
            secondary = (np.asarray(perFrameSecondary, dtype=float) if perFrameSecondary is not None
                         else np.zeros(nFrames))
            angleSource = "per-frame Positioner Position Sequence"
    if primary is None:
        raise ValueError("no per-frame positioner angles (PositionerPrimaryAngleIncrement)")

    frameTimes = _floats(ds.get("FrameTimeVector"))
    sweeps = findSweeps(primary - primary[0], frameTimes if len(frameTimes) == nFrames else None)
    arcDeg = max(float(abs(primary[b - 1] - primary[a])) for a, b in sweeps)
    for a, b in sweeps:
        steps = np.diff(primary[a:b])
        direction = np.sign(primary[b - 1] - primary[a])
        if steps.size and np.any(np.sign(steps[steps != 0]) != direction):
            warnings.append("The primary angle does not change monotonically within a sweep; frames that "
                            "step back are kept but the angular weighting may be off.")
            break
    acquisitionType = " ".join(_text(_privateElement(ds, 0x0019, GE_DL_IMG_CREATOR, offset)) for offset in (0xB1, 0xBB))
    subtractionSpin = "SUB" in acquisitionType.split() or "SUBTRACTION" in _text(ds.get("ImageType")).upper()
    if len(sweeps) > 1:
        notes.append(f"The run holds {len(sweeps)} sweeps over the same arc (frames "
                     + ", ".join(f"{a + 1}-{b}" for a, b in sweeps) + ")"
                     + ("; a subtraction spin: the first sweep is normally the mask, the second the contrast run."
                        if subtractionSpin else "."))

    # --- distances ---
    sid = sod = None
    distanceSource = ""
    geSid = _privateElement(ds, 0x0019, GE_DL_IMG_CREATOR, GE_SID_OFFSET)
    geSod = _privateElement(ds, 0x0019, GE_DL_IMG_CREATOR, GE_SOD_OFFSET)
    if geSid is not None and geSod is not None:
        sidValues = _float32Array(geSid, nFrames)
        sodValues = _float32Array(geSod, nFrames)
        if sidValues and sodValues and min(sidValues) > 100 and min(sodValues) > 100:
            sid, sod = np.asarray(sidValues, float), np.asarray(sodValues, float)
            distanceSource = "GE per-frame private distances (GEMS_DL_IMG_01)"
    if sid is None:
        sidValue = _floats(ds.get("DistanceSourceToDetector")) or _floats(_find(ds, "DistanceSourceToDetector"))
        sodValue = (_floats(ds.get("DistanceSourceToPatient"))
                    or _floats(_find(ds, "DistanceSourceToIsocenter"))
                    or _floats(_find(ds, "DistanceSourceToPatient")))
        perFrameSid = _perFrameValues(ds, "DistanceSourceToDetector", nFrames)
        perFrameSod = _perFrameValues(ds, "DistanceSourceToIsocenter", nFrames)
        if perFrameSid is not None and perFrameSod is not None:
            sid, sod = np.asarray(perFrameSid, float), np.asarray(perFrameSod, float)
            distanceSource = "per-frame X-Ray Geometry Sequence"
        elif sidValue and sodValue:
            sid, sod = np.full(nFrames, sidValue[0]), np.full(nFrames, sodValue[0])
            distanceSource = "DistanceSourceToDetector / DistanceSourceToPatient"
    if sid is None:
        raise ValueError("source-to-detector / source-to-isocenter distances missing")
    if np.any(sod >= sid) or np.any(sod <= 0):
        raise ValueError(f"implausible distances: source-isocenter {sod.min():.0f} mm, "
                         f"source-detector {sid.min():.0f} mm")
    if np.ptp(sid) > 1.0 or np.ptp(sod) > 1.0:
        notes.append(f"Distances change during the run: SID {sid.min():.1f}-{sid.max():.1f} mm, "
                     f"SOD {sod.min():.1f}-{sod.max():.1f} mm (used per frame).")

    # --- detector ---
    spacing = (_floats(ds.get("ImagerPixelSpacing")) or _floats(_find(ds, "ImagerPixelSpacing"))
               or _floats(_find(ds, "DetectorElementSpacing")))
    if len(spacing) < 2:
        pixelSpacing = _floats(ds.get("PixelSpacing"))
        if len(pixelSpacing) >= 2:
            # PixelSpacing of a projection image is at the isocenter plane in some exports
            spacing = [p * float(sid[0] / sod[0]) for p in pixelSpacing[:2]]
            warnings.append("ImagerPixelSpacing missing; detector pixel size derived from PixelSpacing "
                            "assuming it is given at the isocenter.")
        else:
            fov = _floats(ds.get("FieldOfViewDimensions"))
            if len(fov) >= 2:
                spacing = [fov[0] / rows, fov[1] / columns]
                warnings.append("ImagerPixelSpacing missing; detector pixel size derived from the field "
                                "of view dimensions.")
    if len(spacing) < 2:
        raise ValueError("detector pixel spacing (ImagerPixelSpacing) missing")

    collimator = None
    left = _floats(ds.get("CollimatorLeftVerticalEdge"))
    right = _floats(ds.get("CollimatorRightVerticalEdge"))
    upper = _floats(ds.get("CollimatorUpperHorizontalEdge"))
    lower = _floats(ds.get("CollimatorLowerHorizontalEdge"))
    if left and right and upper and lower:
        # Edges are pixel positions (1-based in practice); keep them inside the image
        c0, c1 = int(max(0, left[0] - 1)), int(min(columns - 1, right[0] - 1))
        r0, r1 = int(max(0, upper[0] - 1)), int(min(rows - 1, lower[0] - 1))
        if c1 - c0 > columns // 4 and r1 - r0 > rows // 4:
            collimator = (r0, r1, c0, c1)

    fovRotation = _floats(ds.get("FieldOfViewRotation"))
    fovFlip = _text(ds.get("FieldOfViewHorizontalFlip")).upper() == "YES"

    relationship = _text(ds.get("PixelIntensityRelationship")).upper()
    if relationship == "DISP":
        notes.append("Pixel values are display-processed (PixelIntensityRelationship DISP): the log "
                     "transform is approximate, so attenuation values are relative, not calibrated.")
    elif relationship == "LOG":
        notes.append("Pixel values are already logarithmic (PixelIntensityRelationship LOG).")

    calibration = None
    try:
        calibration = readGeCalibration(ds)
    except Exception as e:
        warnings.append(f"GE 3D calibration present but unreadable ({e}); using the nominal geometry.")
    if calibration is not None:
        notes.append(f"GE 3D calibration found: {len(calibration['matrices'])} projection matrices, "
                     f"calibrated {calibration['date'] or '?'}"
                     + (f", mean error {calibration['meanError']} mm" if calibration.get("meanError") else "") + ".")

    patientPosition = _text(ds.get("PatientPosition")).upper()
    return {
        "nFrames": nFrames,
        "rows": rows,
        "columns": columns,
        "pixelSpacing": (float(spacing[0]), float(spacing[1])),
        "primaryDeg": np.asarray(primary, float),
        "secondaryDeg": np.asarray(secondary, float),
        "sid": np.asarray(sid, float),
        "sod": np.asarray(sod, float),
        "arcDeg": arcDeg,
        "sweeps": sweeps,
        "subtractionSpin": subtractionSpin,
        "calibration": calibration,
        "angleSource": angleSource,
        "distanceSource": distanceSource,
        "collimator": collimator,
        "fovRotation": float(fovRotation[0]) if fovRotation else 0.0,
        "fovFlip": fovFlip,
        "intensityRelationship": relationship,
        "photometric": _text(ds.get("PhotometricInterpretation")).upper(),
        "bitsStored": int(ds.get("BitsStored", 16) or 16),
        "kvp": (_floats(ds.get("KVP")) or [None])[0],
        "manufacturer": _text(ds.get("Manufacturer")),
        "model": _text(ds.get("ManufacturerModelName")),
        "seriesDescription": _text(ds.get("SeriesDescription")),
        "patientPosition": patientPosition,
        "warnings": warnings,
        "notes": notes,
    }


def describeGeometry(geometry):
    """One-paragraph summary for the log."""
    sid, sod = float(np.median(geometry["sid"])), float(np.median(geometry["sod"]))
    rows, columns = geometry["rows"], geometry["columns"]
    du = geometry["pixelSpacing"][1]
    halfFan = np.degrees(np.arctan(0.5 * columns * du / sid))
    primary = geometry["primaryDeg"]
    return (f"{geometry['nFrames']} frames {columns}x{rows}, detector pixel "
            f"{geometry['pixelSpacing'][1]:.3f} x {geometry['pixelSpacing'][0]:.3f} mm, "
            f"SID {sid:.0f} mm, SOD {sod:.0f} mm (magnification {sid / sod:.2f}), "
            f"primary angle {primary[0]:.1f} to {primary[-1]:.1f} deg ({geometry['arcDeg']:.1f} deg arc), "
            f"fan angle {2 * halfFan:.1f} deg - a short scan needs {180 + 2 * halfFan:.1f} deg")


# ---------------------------------------------------------------------------
# Pixels
# ---------------------------------------------------------------------------
def readFrames(path, geometry=None, log=logging.info):
    """All frames as float32 [frames, rows, columns], with the stored field-of-view rotation/flip
    undone so that row 0 / column 0 is the same detector corner in every export."""
    frames = None
    try:
        import pydicom
        ds = pydicom.dcmread(path)
        frames = ds.pixel_array
        slope = float(ds.get("RescaleSlope", 1) or 1)
        intercept = float(ds.get("RescaleIntercept", 0) or 0)
        frames = frames.astype(np.float32) * slope + intercept
    except Exception as e:
        log(f"pydicom could not decode the pixel data ({e}); trying SimpleITK.")
        import SimpleITK as sitk
        frames = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
    if frames.ndim == 2:
        frames = frames[np.newaxis]
    if frames.ndim == 4:  # colour data: keep the first channel
        frames = frames[..., 0]
    if geometry is not None:
        if geometry.get("photometric") == "MONOCHROME1":
            frames = frames.max() - frames
        frames = undoFieldOfViewTransform(frames, geometry.get("fovRotation", 0.0), geometry.get("fovFlip", False))
    return np.ascontiguousarray(frames, dtype=np.float32)


def undoFieldOfViewTransform(frames, rotationDeg, flipped):
    """The stored image was rotated clockwise by FieldOfViewRotation, then flipped horizontally
    when FieldOfViewHorizontalFlip is YES; undo both (flip first, then rotate back)."""
    if flipped:
        frames = frames[:, :, ::-1]
    quarterTurns = int(round(rotationDeg / 90.0)) % 4
    if quarterTurns:
        # np.rot90 with k>0 rotates counter-clockwise, which undoes a clockwise rotation
        frames = np.rot90(frames, k=quarterTurns, axes=(1, 2))
    return frames

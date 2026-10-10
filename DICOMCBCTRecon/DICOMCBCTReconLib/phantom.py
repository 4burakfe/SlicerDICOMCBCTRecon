"""Synthetic test data: an analytic ellipsoid phantom, its exact cone-beam projections and a
synthetic rotational XA DICOM file. Contains no patient data; used by the tests and the self-test.
"""

import numpy as np

# (center x, y, z mm), (semi-axes mm), attenuation 1/mm - added together where they overlap
# Roughly: a water cylinder-like body, two "bones", a contrast-filled "vessel" and low-contrast spots
DEFAULT_PHANTOM = [
    ((0.0, 0.0, 0.0), (70.0, 55.0, 60.0), 0.020),
    ((-35.0, 10.0, 0.0), (10.0, 12.0, 40.0), 0.025),
    ((38.0, -5.0, 5.0), (8.0, 8.0, 30.0), 0.025),
    ((5.0, -20.0, -10.0), (4.0, 4.0, 35.0), 0.030),
    ((0.0, 20.0, 15.0), (12.0, 12.0, 12.0), 0.004),
    ((15.0, 25.0, -20.0), (6.0, 6.0, 6.0), -0.004),
]


def ellipsoidLineIntegrals(sources, pixels, ellipsoids=DEFAULT_PHANTOM):
    """Exact line integrals from each source point to each detector pixel.
    sources [3] or [..., 3], pixels [..., 3] (mm) -> [...] (dimensionless)."""
    sources = np.asarray(sources, float)
    pixels = np.asarray(pixels, float)
    direction = pixels - sources
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    total = np.zeros(pixels.shape[:-1])
    for center, axes, mu in ellipsoids:
        center, axes = np.asarray(center, float), np.asarray(axes, float)
        o = (sources - center) / axes
        d = direction / axes
        a = np.sum(d * d, axis=-1)
        b = 2.0 * np.sum(o * d, axis=-1)
        c = np.sum(o * o, axis=-1) - 1.0
        disc = b * b - 4.0 * a * c
        chord = np.where(disc > 0, np.sqrt(np.maximum(disc, 0.0)) / a, 0.0)
        total += mu * chord
    return total


def projectPhantom(geometry, ellipsoids=DEFAULT_PHANTOM):
    """Exact line integrals [views, rows, cols] for a ProjectionGeometry."""
    u = geometry.columnPositions()
    v = geometry.rowPositions()
    out = np.zeros((geometry.nViews, geometry.rows, geometry.columns), dtype=np.float32)
    for i in range(geometry.nViews):
        pixels = (geometry.detCenter[i] + u[None, :, None] * geometry.eu[i]
                  + v[:, None, None] * geometry.ev[i])
        out[i] = ellipsoidLineIntegrals(geometry.source[i], pixels, ellipsoids)
    return out


def voxelizePhantom(grid, ellipsoids=DEFAULT_PHANTOM):
    """Phantom sampled at voxel centers [nz, ny, nx] (1/mm)."""
    x, y, z = grid.axes()
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    volume = np.zeros(grid.shape, dtype=np.float32)
    for center, axes, mu in ellipsoids:
        inside = (((X - center[0]) / axes[0]) ** 2 + ((Y - center[1]) / axes[1]) ** 2
                  + ((Z - center[2]) / axes[2]) ** 2) <= 1.0
        volume[inside] += mu
    return volume


def syntheticRunHeader(nFrames=181, rows=128, columns=128, pixelMm=2.4, sid=1193.0, sod=720.0,
                       startDeg=-102.5, arcDeg=200.0, secondaryDeg=0.0, gePrivate=True):
    """Geometry dict as xadata.readGeometry returns it (for tests that skip DICOM)."""
    return {
        "nFrames": nFrames, "rows": rows, "columns": columns, "pixelSpacing": (pixelMm, pixelMm),
        "primaryDeg": startDeg + np.linspace(0.0, arcDeg, nFrames),
        "secondaryDeg": np.full(nFrames, secondaryDeg),
        "sid": np.full(nFrames, sid), "sod": np.full(nFrames, sod),
        "arcDeg": arcDeg, "collimator": None, "fovRotation": 0.0, "fovFlip": False,
    }


def writeSyntheticRun(path, geometryDict, lineIntegrals, i0=4000.0, gePrivate=True, seriesUid=None,
                      encoding="LIN", calibration=None, subtractionSpin=False):
    """Write a synthetic multi-frame XA rotational run (12-bit, MONOCHROME2). Intensities are
    I0 exp(-p), stored as they are (encoding LIN) or as their square root scaled to 12 bits (SQRT).
    Several sweeps are written by repeating the angles (increments restart at each sweep).
    `calibration` (as syntheticCalibration returns it) is written to the GEMS_XR3DCAL_01 block.
    All identifiers are synthetic."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    n, rows, cols = lineIntegrals.shape
    intensity = i0 * np.exp(-lineIntegrals)
    if encoding == "SQRT":
        intensity = np.sqrt(intensity / i0) * 4000.0
    intensities = np.clip(np.round(intensity), 0, 4095).astype(np.uint16)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = seriesUid or generate_uid()
    ds.Modality = "XA"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "SINGLE PLANE"]
    ds.PatientName = "Phantom^Synthetic"
    ds.PatientID = "SYNTHETIC"
    ds.SeriesDescription = "Synthetic rotational run"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.Manufacturer = "SYNTHETIC"
    ds.ScanOptions = "ROTA"
    ds.PositionerMotion = "DYNAMIC"
    primary = np.asarray(geometryDict["primaryDeg"], float)
    secondary = np.asarray(geometryDict["secondaryDeg"], float)
    ds.PositionerPrimaryAngle = f"{primary[0]:.4f}"
    # Increments restart at every sweep, as in GE subtraction spins
    sweepStarts = [0] + [i for i in range(1, len(primary)) if abs(primary[i] - primary[i - 1]) > 10.0]
    ds.PositionerSecondaryAngle = f"{secondary[0]:.4f}"
    ds.PositionerPrimaryAngleIncrement = [f"{a:.4f}" for a in primary - primary[0]]
    frameTimes = [0.0 if i in sweepStarts else 33.0 for i in range(len(primary))]
    ds.FrameTimeVector = [f"{t:.0f}" for t in frameTimes]
    ds.PositionerSecondaryAngleIncrement = [f"{a:.4f}" for a in secondary - secondary[0]]
    ds.DistanceSourceToDetector = f"{geometryDict['sid'][0]:.2f}"
    ds.DistanceSourceToPatient = f"{geometryDict['sod'][0]:.2f}"
    ds.ImagerPixelSpacing = [f"{geometryDict['pixelSpacing'][0]:.6f}", f"{geometryDict['pixelSpacing'][1]:.6f}"]
    ds.PatientPosition = "HFS"
    ds.PixelIntensityRelationship = encoding
    ds.FieldOfViewRotation = "0"
    ds.FieldOfViewHorizontalFlip = "NO"
    if gePrivate:
        block = ds.private_block(0x0019, "GEMS_DL_IMG_01", create=True)
        block.add_new(0xBE, "FL", [float(v) for v in geometryDict["sid"]])
        block.add_new(0xE9, "FL", [float(v) for v in geometryDict["sod"]])
        block.add_new(0xB1, "LO", "ANGIO SUB SPIN 3D_PATIENT" if subtractionSpin else "ANGIO NOSUB SPIN 3D_PATIENT")
        block.add_new(0xBB, "CS", "SUB" if subtractionSpin else "NOSUB")
    if calibration is not None:
        cal = ds.private_block(0x0021, "GEMS_XR3DCAL_01", create=True)
        cal.add_new(0x01, "IS", str(calibration["gridSize"][0]))
        cal.add_new(0x02, "IS", str(calibration["gridSize"][1]))
        cal.add_new(0x03, "FL", float(calibration["fovMm"]))
        cal.add_new(0x04, "DA", calibration["date"])
        cal.add_new(0x09, "FL", [float(a) for a in calibration["angles"]])
        items = []
        for P in calibration["matrices"]:
            item = Dataset()
            itemBlock = item.private_block(0x0021, "GEMS_XR3DCAL_01", create=True)
            itemBlock.add_new(0x0C, "DS", [f"{v:.10g}" for v in np.asarray(P).ravel()])
            items.append(item)
        cal.add_new(0x0B, "SQ", items)
        cal.add_new(0x15, "FL", [float(v) for v in calibration["pixelMm"]])
        cal.add_new(0x16, "CS", calibration["kind"])
        cal.add_new(0x20, "LT", "<AugmentedCalib>\n<Gantrymodel>\ngantry mean error : 0.36\n</Gantrymodel>\n</AugmentedCalib>")
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames = n
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = intensities.tobytes()
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    return path


def perturbGeometry(projectionGeometry, shiftU=3.0, shiftV=2.0, sag=2.5):
    """Make a geometry 'real': the detector shifts and the source sags with the gantry angle, as a
    C-arm does. Returns the same object, modified."""
    g = projectionGeometry
    a = np.radians(g.primaryDeg)
    detCenter = (g.detCenter + (shiftU * np.sin(a))[:, None] * g.eu + shiftV * g.ev)
    source = g.source + (sag * np.cos(a))[:, None] * np.array([0.0, 0.0, 1.0])
    g.setVectors(source, detCenter, g.eu, g.ev)
    return g


def syntheticCalibration(header, calibrationAngles, perturbation=None, gridSize=750, pixelMm=None):
    """A GE-style calibration (GEMS_XR3DCAL_01 contents) of the trajectory in `header`: projection
    matrices at the calibration angles, expressed in a calibration frame with x/y swapped, z flipped,
    0.1 mm units and an offset, for a gridSize x gridSize pixel grid covering the same detector."""
    from .geometry import ProjectionGeometry, matrixFromVectors
    calibrationHeader = dict(header, primaryDeg=np.asarray(calibrationAngles, float),
                             secondaryDeg=np.zeros(len(calibrationAngles)),
                             sid=np.full(len(calibrationAngles), header["sid"][0]),
                             sod=np.full(len(calibrationAngles), header["sod"][0]),
                             nFrames=len(calibrationAngles))
    g = ProjectionGeometry(calibrationHeader, {"useCalibration": False})
    if perturbation is not None:
        perturbGeometry(g, **perturbation)
    detectorMm = header["columns"] * header["pixelSpacing"][1]
    pixelMm = pixelMm or detectorMm / gridSize
    # LPS mm -> calibration frame (units 0.1 mm): x_cal = y, y_cal = x, z_cal = -z, plus an offset
    toCal = np.zeros((4, 4))
    toCal[0, 1] = toCal[1, 0] = 10.0
    toCal[2, 2] = -10.0
    toCal[:3, 3] = [150.0, -80.0, 40.0]
    toCal[3, 3] = 1.0
    fromCal = np.linalg.inv(toCal)
    matrices = []
    for i in range(g.nViews):
        P = matrixFromVectors(g.source[i], g.detCenter[i], g.eu[i], g.ev[i], pixelMm, pixelMm, gridSize, gridSize)
        matrices.append(P @ fromCal * 7.3)  # arbitrary overall scale, as stored
    return {"angles": np.asarray(calibrationAngles, float), "matrices": np.asarray(matrices),
            "gridSize": (gridSize, gridSize), "pixelMm": (pixelMm, pixelMm), "fovMm": detectorMm,
            "date": "20260101", "kind": "RECTIFICATION", "meanError": None}

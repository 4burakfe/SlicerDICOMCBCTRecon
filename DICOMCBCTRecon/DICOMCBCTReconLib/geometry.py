"""C-arm projection geometry and the reconstruction grid (numpy only).

Coordinates are DICOM patient coordinates (LPS, mm) with the origin at the isocenter:
+x towards the patient's left, +y posterior, +z superior (head).

Detector model at primary angle a (LAO positive) and secondary angle b (CRA positive):
    d  = unit vector isocenter -> detector = Rz(a) Rx(b) (0, -1, 0)
         (at a = b = 0 the detector is anterior: an AP view with the tube under the table)
    eu = detector column direction (image left -> right) = Rz(a) Rx(b) (1, 0, 0)
         (patient's left at a = 0: the image is seen as when facing the patient)
    ev = detector row direction (image top -> bottom) = eu x d  (towards the feet at a = b = 0)
    source = -SOD d, detector center = (SID - SOD) d + detector offsets
Vendors differ in how they store the image, so the reconstruction window offers mirroring of
the detector columns/rows, an angle offset and a detector shift to correct a mismatch.
"""

import math

import numpy as np


def _rotationZ(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotationX(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


DEFAULT_OPTIONS = {
    "flipColumns": False,      # mirror the detector image left-right
    "flipRows": False,         # mirror the detector image top-bottom
    "negateAngles": False,     # use -angle (mirror image of the trajectory)
    "angleOffsetDeg": 0.0,     # added to every primary angle
    "offsetU": 0.0,            # detector shift along the columns direction (mm, at the detector)
    "offsetV": 0.0,            # detector shift along the rows direction (mm, at the detector)
    "binning": 1,              # detector pixel binning (1, 2, 4)
    "useCalibration": True,    # use calibrated projection matrices when the run carries them
}


def vectorsFromMatrix(P, du, rows, cols):
    """Source, detector center and axes (LPS mm) of a 3x4 projection matrix that maps LPS mm to
    (column, row) pixels of a rows x cols detector with column spacing du.
    Returns (source, detCenter, eu, ev, rowSpacing) - the row spacing implied by the matrix."""
    from .gecalib import decompose
    C, K, R = decompose(P)
    fu, fv, u0, v0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    sdd = fu * du
    eu, ev, axis = R[0], R[1], R[2]
    dv = sdd / fv
    center = C + sdd * axis + (0.5 * (cols - 1) - u0) * du * eu + (0.5 * (rows - 1) - v0) * dv * ev
    return C, center, eu, ev, dv


class ProjectionGeometry:
    """Per-view source and detector vectors for the (binned, possibly mirrored) frames.

    Attributes (numpy, one row per view):
        source, detCenter, eu, ev   [n, 3]  mm / unit vectors
        normal                      [n, 3]  unit detector normal pointing away from the source
        sdd                         [n]     source to detector plane distance (mm)
        sod                         [n]     source to isocenter distance along the normal (mm)
        u0, v0                      [n]     principal point on the detector (mm from its center)
        beta                        [n]     rotation angle along the trajectory, starting at 0 (rad)
        dBeta                       [n]     angular interval covered by each view (rad)
    and the detector size (rows, columns) and pixel size (du = column spacing, dv = row spacing)."""

    def __init__(self, geometry, options=None):
        options = dict(DEFAULT_OPTIONS, **(options or {}))
        self.options = options
        binning = max(1, int(options["binning"]))
        self.binning = binning
        self.rows = geometry["rows"] // binning
        self.columns = geometry["columns"] // binning
        self.dv = geometry["pixelSpacing"][0] * binning
        self.du = geometry["pixelSpacing"][1] * binning

        primary = np.asarray(geometry["primaryDeg"], float) + float(options["angleOffsetDeg"])
        secondary = np.asarray(geometry["secondaryDeg"], float)
        if options["negateAngles"]:
            primary, secondary = -primary, -secondary
        sid = np.asarray(geometry["sid"], float)
        sod = np.asarray(geometry["sod"], float)
        n = primary.size
        self.nViews = n
        self.primaryDeg = primary

        source = np.zeros((n, 3))
        detCenter = np.zeros((n, 3))
        eu = np.zeros((n, 3))
        ev = np.zeros((n, 3))
        self.calibrated = False
        self.calibrationInfo = None
        matrices = None
        if options["useCalibration"] and geometry.get("calibration") is not None:
            from . import gecalib
            calibrationGeometry = dict(geometry, primaryDeg=np.asarray(geometry["primaryDeg"], float)
                                       + float(options["angleOffsetDeg"]))
            matrices, self.calibrationInfo = gecalib.frameMatrices(calibrationGeometry)
        if matrices is not None:
            rowSpacings = []
            for i in range(n):
                source[i], center, eu[i], ev[i], dvImplied = vectorsFromMatrix(
                    matrices[i], geometry["pixelSpacing"][1], geometry["rows"], geometry["columns"])
                detCenter[i] = center + options["offsetU"] * eu[i] + options["offsetV"] * ev[i]
                rowSpacings.append(dvImplied)
            # The matrices fix the pixel aspect; keep it (it differs from 1 by a fraction of a percent)
            self.dv = float(np.median(rowSpacings)) * binning
            self.calibrated = True
            if options["negateAngles"]:
                mirror = np.diag([-1.0, 1.0, 1.0])  # mirror image across the sagittal plane
                source, detCenter, eu, ev = source @ mirror, detCenter @ mirror, eu @ mirror, ev @ mirror
        else:
            for i in range(n):
                rotation = _rotationZ(math.radians(primary[i])) @ _rotationX(math.radians(secondary[i]))
                d = rotation @ np.array([0.0, -1.0, 0.0])
                u = rotation @ np.array([1.0, 0.0, 0.0])
                v = np.cross(u, d)
                source[i] = -sod[i] * d
                detCenter[i] = (sid[i] - sod[i]) * d + options["offsetU"] * u + options["offsetV"] * v
                eu[i], ev[i] = u, v
        if options["flipColumns"]:
            eu = -eu
        if options["flipRows"]:
            ev = -ev
        self.setVectors(source, detCenter, eu, ev)

    def setVectors(self, source, detCenter, eu, ev):
        """Set the per-view source/detector vectors and derive everything else from them."""
        n = len(source)
        normal = np.cross(ev, eu)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        # Orient the normal from the source towards the detector
        sign = np.sign(np.einsum("ij,ij->i", detCenter - source, normal))
        sign[sign == 0] = 1.0
        normal *= sign[:, None]
        self.source, self.detCenter, self.eu, self.ev, self.normal = source, detCenter, eu, ev, normal
        self.sdd = np.einsum("ij,ij->i", detCenter - source, normal)
        self.sod = np.einsum("ij,ij->i", -source, normal)
        principal = source + self.sdd[:, None] * normal
        self.u0 = np.einsum("ij,ij->i", principal - detCenter, eu)
        self.v0 = np.einsum("ij,ij->i", principal - detCenter, ev)

        # Angle travelled along the trajectory (handles either rotation direction)
        directions = -source / np.linalg.norm(source, axis=1, keepdims=True)
        steps = np.zeros(n)
        for i in range(1, n):
            steps[i] = math.acos(float(np.clip(np.dot(directions[i - 1], directions[i]), -1.0, 1.0)))
        self.beta = np.cumsum(steps)
        dBeta = np.zeros(n)
        if n > 1:
            dBeta[1:-1] = 0.5 * (self.beta[2:] - self.beta[:-2])
            dBeta[0] = 0.5 * (self.beta[1] - self.beta[0])
            dBeta[-1] = 0.5 * (self.beta[-1] - self.beta[-2])
        self.dBeta = dBeta
        # Scan range including half a step at each end
        self.scanRange = float(self.beta[-1] + dBeta[0] + dBeta[-1]) if n > 1 else 0.0

        # Rotation sense relative to the detector columns: +1 when the source moves along +eu
        if n > 1:
            motion = source[min(n - 1, n // 2 + 1)] - source[max(0, n // 2 - 1)]
            self.rotationSign = 1.0 if np.dot(motion, eu[n // 2]) >= 0 else -1.0
        else:
            self.rotationSign = 1.0

    # --- detector coordinates ---
    def columnPositions(self):
        """Column centers in mm from the detector center (along eu)."""
        return (np.arange(self.columns) - 0.5 * (self.columns - 1)) * self.du

    def rowPositions(self):
        return (np.arange(self.rows) - 0.5 * (self.rows - 1)) * self.dv

    def halfFanAngle(self):
        """Half fan angle (rad) of the detector, about the principal ray."""
        halfWidth = 0.5 * self.columns * self.du
        return float(np.max(np.arctan((halfWidth + np.abs(self.u0)) / self.sdd)))

    def shortScanRange(self):
        return math.pi + 2.0 * self.halfFanAngle()

    def fovRadius(self):
        """Radius (mm) of the cylinder about the rotation axis seen by every view."""
        halfWidth = 0.5 * self.columns * self.du
        reach = np.minimum(np.abs(-halfWidth - self.u0), np.abs(halfWidth - self.u0))
        angle = np.arctan(reach / self.sdd)
        return float(np.min(self.sod * np.sin(angle)))

    def fovHeight(self):
        """Axial extent (mm) of the detector at the isocenter."""
        return float(np.min(self.rows * self.dv * self.sod / self.sdd))

    def projectPoints(self, points, view):
        """Detector (column, row) pixel indices of LPS points [N, 3] in one view (for tests)."""
        rel = points - self.source[view]
        depth = rel @ self.normal[view]
        hit = self.source[view] + rel * (self.sdd[view] / depth)[:, None]
        u = (hit - self.detCenter[view]) @ self.eu[view]
        v = (hit - self.detCenter[view]) @ self.ev[view]
        return u / self.du + 0.5 * (self.columns - 1), v / self.dv + 0.5 * (self.rows - 1)


class ReconGrid:
    """Axis-aligned LPS voxel grid: shape (nz, ny, nx) as numpy arrays index it, spacing (sx, sy, sz),
    origin = LPS position of voxel (0, 0, 0)."""

    def __init__(self, size, spacing, center=(0.0, 0.0, 0.0)):
        self.nx, self.ny, self.nz = (int(s) for s in size)
        self.sx, self.sy, self.sz = (float(s) for s in spacing)
        center = np.asarray(center, float)
        self.origin = center - 0.5 * np.array([(self.nx - 1) * self.sx, (self.ny - 1) * self.sy,
                                               (self.nz - 1) * self.sz])

    @property
    def shape(self):
        return (self.nz, self.ny, self.nx)

    def axes(self):
        """1D coordinates (x, y, z) of voxel centers."""
        return (self.origin[0] + np.arange(self.nx) * self.sx,
                self.origin[1] + np.arange(self.ny) * self.sy,
                self.origin[2] + np.arange(self.nz) * self.sz)

    def extent(self):
        return np.array([(self.nx - 1) * self.sx, (self.ny - 1) * self.sy, (self.nz - 1) * self.sz])

    def key(self):
        return (self.nx, self.ny, self.nz, round(self.sx, 4), round(self.sy, 4), round(self.sz, 4),
                tuple(np.round(self.origin, 3)))


def defaultGrid(projectionGeometry, matrix=256, voxelMm=None, maxSlices=None):
    """Grid that covers the field of view seen by all views: a matrix x matrix axial plane spanning the
    FOV diameter and as many slices as the cone covers at the isocenter."""
    radius = projectionGeometry.fovRadius()
    height = projectionGeometry.fovHeight()
    if voxelMm is None or voxelMm <= 0:
        voxelMm = 2.0 * radius / matrix
    slices = int(round(height / voxelMm))
    if maxSlices:
        slices = min(slices, maxSlices)
    slices = max(1, slices)
    return ReconGrid((matrix, matrix, slices), (voxelMm, voxelMm, voxelMm))


def matrixFromVectors(source, detCenter, eu, ev, du, dv, rows, cols):
    """3x4 projection matrix (LPS mm -> column, row pixels) of a detector given by its center and
    unit axes; the inverse of vectorsFromMatrix (used by the tests and the phantom)."""
    normal = np.cross(eu, ev)
    normal /= np.linalg.norm(normal)
    if np.dot(detCenter - source, normal) < 0:
        normal = -normal
    rel = detCenter - source
    sdd, pu, pv = float(rel @ normal), float(rel @ eu), float(rel @ ev)
    K = np.array([[sdd / du, 0.0, 0.5 * (cols - 1) - pu / du],
                  [0.0, sdd / dv, 0.5 * (rows - 1) - pv / dv],
                  [0.0, 0.0, 1.0]])
    Rt = np.vstack([eu, ev, normal])
    return K @ np.hstack([Rt, (-Rt @ source)[:, None]])

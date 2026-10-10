"""GE Innova 3D calibration -> per-frame projection matrices in patient coordinates (numpy only).

GE stores, in the private block GEMS_XR3DCAL_01, one projection matrix per calibration position,
measured on a phantom. These matrices include the C-arm's real (non-ideal) motion: sag, wobble,
detector shift and tilt, which the nominal angles in the DICOM header do not describe.

The calibration frame is GE's own (units of 0.1 mm, axes not aligned with the patient). It is tied
to patient coordinates (LPS, mm, isocenter at the origin) by fitting the calibrated source positions
to the nominal ones of the same gantry angles with a similarity transform (rotation, scale,
translation). Matrices are then re-expressed for the acquisition's pixel grid and interpolated to
the gantry angle of every frame.
"""

import math

import numpy as np

# A rigid fit worse than this (RMS distance of source positions, mm) means the calibration does not
# match the nominal trajectory, so it is not used
MAX_FIT_RMS_MM = 25.0
# Allowed difference between the fitted scale and a round unit (0.01, 0.1 or 1 mm)
MAX_SCALE_DEVIATION = 0.05


def normalizeMatrix(P):
    """Scale a projection matrix so that its third row has a unit direction part and the origin
    (near the isocenter) lies in front of the source (positive depth)."""
    P = np.asarray(P, dtype=float)
    scale = np.linalg.norm(P[2, :3])
    if P[2, 3] < 0:
        scale = -scale
    return P / scale


def decompose(P):
    """P = K [R | -R C]. Returns (C source position, K intrinsics with positive diagonal, R rows =
    directions of increasing column, increasing row, and the principal ray)."""
    P = normalizeMatrix(P)
    M = P[:, :3]
    C = -np.linalg.solve(M, P[:, 3])
    # RQ decomposition via QR of the row/column reversed matrix
    flip = np.eye(3)[::-1]
    q, r = np.linalg.qr((flip @ M).T)
    K = flip @ r.T @ flip
    R = flip @ q.T
    signs = np.diag(np.sign(np.diag(K)))
    K = K @ signs
    R = signs @ R
    # R may be improper when the stored image is mirrored; R[2] always points into the scene
    return C, K / K[2, 2], R


def nominalSources(anglesDeg, sod, secondaryDeg=0.0):
    """Source positions (LPS mm) of the nominal model in geometry.py at these primary angles."""
    a = np.radians(np.asarray(anglesDeg, dtype=float))
    b = math.radians(secondaryDeg)
    d = np.stack([np.sin(a) * math.cos(b), -np.cos(a) * math.cos(b), np.full_like(a, math.sin(b))], axis=1)
    return -np.asarray(sod, dtype=float).reshape(-1, 1) * d


def similarityFit(source, target, allowReflection=False):
    """Umeyama: scale s, rotation R, translation t minimising |s R source + t - target|.
    Returns (s, R, t, rms)."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    muS, muT = source.mean(axis=0), target.mean(axis=0)
    a, b = source - muS, target - muT
    u, sigma, vt = np.linalg.svd(b.T @ a / len(a))
    d = np.ones(3)
    if not allowReflection and np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    R = u @ np.diag(d) @ vt
    variance = (a ** 2).sum() / len(a)
    s = float((sigma * d).sum() / variance)
    t = muT - s * R @ muS
    rms = float(np.sqrt((((s * (R @ source.T)).T + t - target) ** 2).sum(axis=1).mean()))
    return s, R, t, rms


def frameMatrices(geometry, log=print):
    """Calibrated projection matrices [frames, 3, 4] for the frames of `geometry` (as returned by
    xadata.readGeometry or sweepGeometry), mapping LPS mm to (column, row) pixels of the stored
    frames. Returns (matrices, info) or (None, info) when the calibration cannot be used."""
    calibration = geometry.get("calibration")
    info = {}
    if calibration is None:
        return None, {"reason": "no calibration"}
    angles = calibration["angles"]
    raw = np.asarray([normalizeMatrix(P) for P in calibration["matrices"]])
    sources = np.asarray([decompose(P)[0] for P in raw])
    sod = float(np.median(geometry["sod"]))
    nominal = nominalSources(angles, np.full(len(angles), sod))
    s, R, t, rms = similarityFit(sources, nominal)
    reflected = False
    if rms > MAX_FIT_RMS_MM:
        s2, R2, t2, rms2 = similarityFit(sources, nominal, allowReflection=True)
        if rms2 < rms:
            s, R, t, rms, reflected = s2, R2, t2, rms2, True
    info.update(rms=rms, scale=s, reflected=reflected)
    if rms > MAX_FIT_RMS_MM:
        info["reason"] = f"calibrated trajectory does not match the header angles (RMS {rms:.1f} mm)"
        return None, info
    # GE calibrations are in 0.1 mm; a scale far from any plausible unit means the header distances
    # and the calibration disagree
    if not any(abs(s / unit - 1.0) < MAX_SCALE_DEVIATION for unit in (0.01, 0.1, 1.0)):
        info["reason"] = (f"calibration scale {s:.4f} mm per unit does not match the header's "
                          f"source-isocenter distance ({sod:.0f} mm)")
        return None, info

    # Calibration frame -> LPS: X_lps = s R X_cal + t, so P_lps = P_cal [R^T / s, -R^T t / s; 0 1]
    toCal = np.eye(4)
    toCal[:3, :3] = R.T / s
    toCal[:3, 3] = -R.T @ t / s

    # Calibration pixel grid -> stored frame pixel grid (both centred on the detector)
    calRows, calCols = calibration["gridSize"]
    calDv, calDu = calibration["pixelMm"]
    dv, du = geometry["pixelSpacing"]
    rows, cols = geometry["rows"], geometry["columns"]
    pixels = np.array([[calDu / du, 0.0, 0.5 * (cols - 1) - 0.5 * (calCols - 1) * calDu / du],
                       [0.0, calDv / dv, 0.5 * (rows - 1) - 0.5 * (calRows - 1) * calDv / dv],
                       [0.0, 0.0, 1.0]])
    fovFrames = cols * du
    if calibration.get("fovMm") and abs(calibration["fovMm"] - fovFrames) > 2.0:
        info["note"] = (f"calibration field of view {calibration['fovMm']:.0f} mm differs from the frames' "
                        f"{fovFrames:.0f} mm; assuming both are centred on the detector")
    mapped = np.asarray([normalizeMatrix(pixels @ P @ toCal) for P in raw])

    # Interpolate to the gantry angle of every frame
    frameAngles = np.asarray(geometry["primaryDeg"], dtype=float)
    outside = int(np.sum((frameAngles < angles[0] - 0.5) | (frameAngles > angles[-1] + 0.5)))
    if outside:
        info["outside"] = outside
    flat = mapped.reshape(len(angles), 12)
    perFrame = np.stack([np.interp(frameAngles, angles, flat[:, k]) for k in range(12)], axis=1)
    matrices = np.asarray([normalizeMatrix(P) for P in perFrame.reshape(-1, 3, 4)])
    return matrices, info


def describe(info, calibration):
    parts = [f"GE 3D calibration ({calibration.get('date') or 'date ?'}): source positions match the header "
             f"angles within {info['rms']:.1f} mm RMS (scale {info['scale']:.4f} mm per unit"
             + (", mirrored frame" if info.get("reflected") else "") + ")"]
    if info.get("outside"):
        parts.append(f"{info['outside']} frames lie outside the calibrated angles (nearest matrix used)")
    if info.get("note"):
        parts.append(info["note"])
    return "; ".join(parts) + "."

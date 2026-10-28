"""Cone-beam reconstruction for C-arm rotational runs (PyTorch, no Slicer).

    lineIntegrals   detector intensities -> line integrals  p = ln(I0 / I)
    fdk             Feldkamp-Davis-Kress filtered backprojection with Parker short-scan weights (or
                    displaced-detector weights for an offset detector over a full rotation), cosine
                    pre-weighting, apodised ramp filter, optional vertical Hann window and RTK-style
                    lateral truncation correction
    forwardProject  ray-driven projector (trilinear sampling), used by the iterative refinement
    sart            ordered-subsets SART (simultaneous algebraic reconstruction) with non-negativity

Units: distances in mm, so reconstructed attenuation is in 1/mm (shown as 1/cm).
torch is imported when used, so importing this module is cheap.
"""

import math

import numpy as np


FILTERS = ["Ram-Lak", "Shepp-Logan", "Cosine", "Hann"]
DEFAULT_FILTER = "Shepp-Logan"

# Voxels seen by fewer than this fraction of the scan's angular range are set to zero
FOV_COVERAGE = 0.98


class ReconstructionAborted(Exception):
    """Raised between views when the user presses Stop."""


def torchDevice(preferGpu=True):
    import torch
    if preferGpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _check(progress, shouldAbort, label, done, total, text):
    if progress is not None:
        progress(label, done, total, text)
    if shouldAbort is not None and shouldAbort():
        raise ReconstructionAborted()


# ---------------------------------------------------------------------------
# Preprocessing (numpy)
# ---------------------------------------------------------------------------
INTENSITY_LOG = "Linear intensities: ln(I0 / I)"
INTENSITY_SQRT = "Square-root encoded: 2 ln(I0 / I)"
INTENSITY_LINEAR = "Already logarithmic: (I0 - I) / I0"
INTENSITY_MODES = [INTENSITY_SQRT, INTENSITY_LOG, INTENSITY_LINEAR]


def defaultIntensityMode(geometry):
    """Pixel encoding from PixelIntensityRelationship. GE stores raw rotational frames square-root
    encoded (SQRT); display-processed GE exports (DISP) are assumed to keep that encoding."""
    relationship = geometry.get("intensityRelationship", "")
    if relationship == "SQRT":
        return INTENSITY_SQRT
    if relationship == "LOG":
        return INTENSITY_LINEAR
    if relationship == "DISP" and "GE" in geometry.get("manufacturer", "").upper():
        return INTENSITY_SQRT
    return INTENSITY_LOG

I0_GLOBAL = "Brightest pixels of the run"
I0_FRAME = "Brightest pixels of each frame"
I0_MANUAL = "Fixed value"
I0_MODES = [I0_GLOBAL, I0_FRAME, I0_MANUAL]
I0_PERCENTILE = 99.9


def binFrames(frames, binning):
    """Average binning x binning detector pixels (frames [n, rows, cols])."""
    if binning <= 1:
        return frames
    n, rows, cols = frames.shape
    rows, cols = rows // binning, cols // binning
    trimmed = frames[:, :rows * binning, :cols * binning]
    return trimmed.reshape(n, rows, binning, cols, binning).mean(axis=(2, 4), dtype=np.float32)


def estimateI0(frames, mode, manualValue=None, collimator=None):
    """Unattenuated intensity: a scalar (global/manual) or one value per frame."""
    if mode == I0_MANUAL and manualValue:
        return float(manualValue)
    region = frames
    if collimator is not None:
        r0, r1, c0, c1 = collimator
        region = frames[:, r0:r1 + 1, c0:c1 + 1]
    if mode == I0_FRAME:
        flat = region.reshape(region.shape[0], -1)
        return np.percentile(flat, I0_PERCENTILE, axis=1).astype(np.float32)
    # Subsample: the percentile of a few million pixels is as good as of all of them
    flat = region.reshape(-1)
    step = max(1, flat.size // 4_000_000)
    return float(np.percentile(flat[::step], I0_PERCENTILE))


def lineIntegrals(frames, intensityMode=INTENSITY_LOG, i0=None, collimator=None):
    """Detector frames [n, rows, cols] -> line integrals (float32). Pixels outside the collimated
    area are replaced by the nearest collimated row/column, so they neither add edges nor
    count as unattenuated air."""
    frames = np.asarray(frames, dtype=np.float32)
    if i0 is None:
        i0 = estimateI0(frames, I0_GLOBAL, collimator=collimator)
    i0Array = np.asarray(i0, dtype=np.float32)
    if i0Array.ndim == 1:
        i0Array = i0Array[:, None, None]
    if intensityMode == INTENSITY_LINEAR:
        p = (i0Array - frames) / np.maximum(i0Array, 1e-6)
    else:
        floor = np.maximum(i0Array * 1e-4, 1e-3)
        p = np.log(i0Array / np.maximum(frames, floor))
        if intensityMode == INTENSITY_SQRT:
            p *= 2.0  # pixel = sqrt(intensity): ln(I0 / I) = 2 ln(pixel0 / pixel)
    p = np.maximum(p, 0.0).astype(np.float32)
    if collimator is not None:
        r0, r1, c0, c1 = collimator
        if c0 > 0:
            p[:, :, :c0] = p[:, :, c0:c0 + 1]
        if c1 < p.shape[2] - 1:
            p[:, :, c1 + 1:] = p[:, :, c1:c1 + 1]
        if r0 > 0:
            p[:, :r0, :] = p[:, r0:r0 + 1, :]
        if r1 < p.shape[1] - 1:
            p[:, r1 + 1:, :] = p[:, r1:r1 + 1, :]
    return p


def binnedCollimator(collimator, binning, rows, cols):
    if collimator is None:
        return None
    r0, r1, c0, c1 = collimator
    b = max(1, binning)
    return (int(math.ceil(r0 / b)), min(rows - 1, (r1 + 1) // b - 1),
            int(math.ceil(c0 / b)), min(cols - 1, (c1 + 1) // b - 1))


# ---------------------------------------------------------------------------
# Weights and filter
# ---------------------------------------------------------------------------
def parkerWeights(geometry):
    """Short-scan redundancy weights [views, columns] (Parker 1982, generalised to overscan by
    using delta = (scan range - pi) / 2). Returns all 0.5 for a full rotation."""
    beta = geometry.beta - geometry.beta[0] + geometry.dBeta[0]
    scan = geometry.scanRange
    nViews, nCols = geometry.nViews, geometry.columns
    if scan >= 2 * math.pi - 1e-3:
        return np.full((nViews, nCols), 0.5, dtype=np.float32)
    delta = max(1e-3, 0.5 * (scan - math.pi))
    u = geometry.columnPositions()[None, :] - geometry.u0[:, None]
    # Fan angle signed so that it is positive towards where the source has come from
    gamma = -geometry.rotationSign * np.arctan(u / geometry.sdd[:, None])
    # Rays outside +-delta have no complete set of conjugates; clamping keeps the weights smooth
    gamma = np.clip(gamma, -0.999 * delta, 0.999 * delta)
    b = beta[:, None]
    w = np.ones((nViews, nCols))
    rise = b < 2 * (delta - gamma)
    w = np.where(rise, np.sin(0.25 * math.pi * b / (delta - gamma)) ** 2, w)
    fall = b > math.pi - 2 * gamma
    w = np.where(fall, np.sin(0.25 * math.pi * (math.pi + 2 * delta - b) / (delta + gamma)) ** 2, w)
    w = np.clip(w, 0.0, 1.0)
    return w.astype(np.float32)


def displacedDetectorWeights(geometry, taper=None):
    """Redundancy weights [views, columns] for a full rotation with a laterally displaced detector
    (half-fan / offset detector, Wang 2002). Rays in the central band of half width `a` (seen from
    both sides of the rotation) get a smooth sin^2 ramp so that conjugate rays sum to one; rays only
    seen once (the wide side) get weight 1. A centred detector gives 0.5 everywhere, like Parker
    weights of a full rotation. `taper` limits the ramp width (mm at the detector)."""
    nViews, nCols = geometry.nViews, geometry.columns
    halfWidth = 0.5 * nCols * geometry.du
    u = geometry.columnPositions()[None, :] - geometry.u0[:, None]          # [views, cols] mm
    left = -halfWidth - geometry.u0                                          # detector edges about the
    right = halfWidth - geometry.u0                                          # principal ray
    a = np.minimum(np.abs(left), np.abs(right))[:, None]                     # overlap half width
    wide = np.where(np.abs(right) >= np.abs(left), 1.0, -1.0)[:, None]       # side that extends further
    if taper is not None:
        a = np.minimum(a, taper)
    if np.all(np.abs(np.abs(right) - np.abs(left)) < 2 * geometry.du):
        return np.full((nViews, nCols), 0.5, dtype=np.float32)
    s = np.clip(wide * u / np.maximum(a, 1e-6), -1.0, 1.0)                   # -1 short edge .. +1
    w = np.sin(0.25 * math.pi * (s + 1.0)) ** 2
    return w.astype(np.float32)


def isDisplacedDetector(geometry, tolerance=0.05):
    """True when the detector is offset by more than `tolerance` of its half width."""
    halfWidth = 0.5 * geometry.columns * geometry.du
    return bool(np.median(np.abs(geometry.u0)) > tolerance * halfWidth)


def isFullRotation(geometry):
    return geometry.scanRange >= 2 * math.pi - math.radians(5.0)


def redundancyWeights(geometry, log=print):
    """Parker weights for a short scan, displaced-detector weights for a full rotation with an offset
    detector (both together are not supported: Parker weights are used with a warning)."""
    if isDisplacedDetector(geometry):
        if isFullRotation(geometry):
            return displacedDetectorWeights(geometry)
        log("WARNING: displaced detector with a short scan; using Parker weights only.")
    return parkerWeights(geometry)


def rampKernel(nPad, tau, filterName=DEFAULT_FILTER, cutoff=1.0):
    """Frequency response (rfft, length nPad//2+1) of the band-limited ramp filter built from its
    spatial form (Kak & Slaney), so the DC term is correct, times an apodisation window.
    cutoff = fraction of the Nyquist frequency where the window reaches zero."""
    n = np.arange(nPad)
    n = np.where(n > nPad // 2, n - nPad, n)
    h = np.zeros(nPad)
    h[n == 0] = 1.0 / (4.0 * tau * tau)
    odd = (n % 2) == 1
    h[odd] = -1.0 / (math.pi * math.pi * tau * tau * n[odd].astype(float) ** 2)
    response = np.real(np.fft.rfft(h)) * tau  # x tau: discrete convolution approximates the integral
    f = np.fft.rfftfreq(nPad)  # cycles per sample, 0..0.5
    fc = 0.5 * max(1e-3, min(1.0, cutoff))
    x = f / fc
    if filterName == "Shepp-Logan":
        window = np.sinc(x / 2.0)
    elif filterName == "Cosine":
        window = np.cos(0.5 * math.pi * np.clip(x, 0, 1))
    elif filterName == "Hann":
        window = 0.5 * (1.0 + np.cos(math.pi * np.clip(x, 0, 1)))
    else:
        window = np.ones_like(f)
    window = np.where(x > 1.0, 0.0, window)
    return (response * window).astype(np.float32)


def _extendRows(p, extension):
    """Lateral truncation correction as in RTK's ramp filter: each row is continued past both edges
    by its mirror image reflected about the edge value (2 p_edge - p_mirror), faded out with
    sin^0.75 weights. Unlike RTK the extension is clamped at zero (line integrals are not negative):
    with an untruncated object the edge is air and the reflection would add a negative copy of the
    object, raising the reconstructed level by a few percent. p: torch [rows, cols]."""
    import torch
    n = min(int(extension), p.shape[-1] - 1)
    if n < 2:
        return p
    i = torch.arange(n, device=p.device, dtype=p.dtype)
    w = torch.sin((n - i) * math.pi / (2 * n - 2)) ** 0.75
    left = (w * (2 * p[..., :1] - p[..., 1:n + 1]).clamp(min=0.0)).flip(-1)
    right = w * (2 * p[..., -1:] - p[..., -n - 1:-1].flip(-1)).clamp(min=0.0)
    return torch.cat([left, p, right], dim=-1)


def _hannWindow(n, cutoff, device):
    """Hann window over rfft frequencies of length n; cutoff = fraction of Nyquist."""
    import torch
    f = torch.fft.rfftfreq(n, device=device)
    x = f / (0.5 * max(1e-3, min(1.0, cutoff)))
    return torch.where(x > 1.0, torch.zeros_like(x), 0.5 * (1.0 + torch.cos(math.pi * x.clamp(0, 1))))


def paddedGeometry(geometry, padLeft, padRight):
    """Copy of the geometry with the detector widened by padLeft / padRight columns (eu side -/+)."""
    import copy
    if not padLeft and not padRight:
        return geometry
    g = copy.copy(geometry)
    g.columns = geometry.columns + padLeft + padRight
    shift = 0.5 * (padRight - padLeft) * geometry.du
    g.setVectors(geometry.source, geometry.detCenter + shift * geometry.eu, geometry.eu, geometry.ev)
    return g


def filterProjections(p, geometry, filterName=DEFAULT_FILTER, cutoff=1.0, truncationExtension=0.25,
                      weights=None, device=None, progress=None, shouldAbort=None, batch=16, cutoffV=None):
    """Cosine pre-weighting, redundancy weights and ramp filtering along the detector rows, done the
    way RTK's FDK does it: the redundancy weights are applied before the ramp filter, a displaced
    detector is zero-padded on its short side to be symmetric about the central ray, then rows are
    extended for truncation correction (fraction of the padded width) and filtered.
    cutoffV, when given, adds a Hann window along the detector's vertical axis (RTK HannCutFrequencyY).
    p: numpy [views, rows, cols] line integrals. Returns (q, filteredGeometry): q is a torch tensor
    on `device`, already multiplied by the angular interval of each view, ready for backprojection
    with filteredGeometry. For a displaced detector q keeps the zero-padded columns: their filtered
    values (the ramp's negative tails) are needed, dropping them biases the periphery upwards."""
    import torch
    device = device or torchDevice()
    nViews, rows, cols = p.shape
    if weights is None:
        weights = redundancyWeights(geometry)
    padLeft = padRight = 0
    if isDisplacedDetector(geometry) and isFullRotation(geometry):
        u0 = float(np.median(geometry.u0))
        padCols = int(round(2.0 * abs(u0) / geometry.du))
        if u0 > 0:      # principal ray right of centre: the left part is the wide side
            padRight = padCols
        else:
            padLeft = padCols
    width = cols + padLeft + padRight
    extension = int(round(truncationExtension * width)) if truncationExtension else 0
    extension = min(extension, width - 1) if extension >= 2 else 0
    nTotal = width + 2 * extension
    nPad = 1 << int(math.ceil(math.log2(2 * nTotal)))
    tauDetector = geometry.du
    windowV = None
    if cutoffV is not None:
        nPadV = 1 << int(math.ceil(math.log2(2 * rows)))
        windowV = _hannWindow(nPadV, cutoffV, device)[:, None]
    u = torch.as_tensor(geometry.columnPositions(), dtype=torch.float32, device=device)
    v = torch.as_tensor(geometry.rowPositions(), dtype=torch.float32, device=device)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    out = torch.empty((nViews, rows, width), dtype=torch.float32, device=device)
    kernels = {}
    for start in range(0, nViews, batch):
        stop = min(nViews, start + batch)
        _check(progress, shouldAbort, "Filtering", start, nViews, f"Filtering projections {start + 1}-{stop} of {nViews}")
        block = torch.as_tensor(p[start:stop], dtype=torch.float32, device=device)
        for k, i in enumerate(range(start, stop)):
            sdd, sod = float(geometry.sdd[i]), float(geometry.sod[i])
            uu = (u - float(geometry.u0[i]))[None, :]
            vv = (v - float(geometry.v0[i]))[:, None]
            cosine = sdd / torch.sqrt(sdd * sdd + uu * uu + vv * vv)
            row = block[k] * cosine * weights[i][None, :]
            if padLeft or padRight:
                row = torch.nn.functional.pad(row, (padLeft, padRight))
            row = _extendRows(row, extension)
            # Filter on the virtual detector through the isocenter (pixel size du * SOD / SDD)
            tau = tauDetector * sod / sdd
            key = round(tau, 6)
            if key not in kernels:
                kernels[key] = torch.as_tensor(rampKernel(nPad, tau, filterName, cutoff), device=device)
            spectrum = torch.fft.rfft(row, n=nPad, dim=-1) * kernels[key]
            filtered = torch.fft.irfft(spectrum, n=nPad, dim=-1)[:, extension:extension + width]
            if windowV is not None:
                nV = windowV.shape[0] * 2 - 2
                filtered = torch.fft.irfft(torch.fft.rfft(filtered, n=nV, dim=0) * windowV, n=nV, dim=0)[:rows]
            out[i] = filtered * float(geometry.dBeta[i])
    return out, paddedGeometry(geometry, padLeft, padRight)


# ---------------------------------------------------------------------------
# Backprojection
# ---------------------------------------------------------------------------
def _sliceChunks(grid, maxVoxels):
    perSlice = grid.nx * grid.ny
    step = max(1, maxVoxels // perSlice)
    return [(z, min(grid.nz, z + step)) for z in range(0, grid.nz, step)]


def backproject(q, geometry, grid, fdkWeighting=True, views=None, device=None,
                progress=None, shouldAbort=None, label="Backprojecting", maxVoxels=8_000_000,
                coverage=None):
    """Voxel-driven backprojection with bilinear detector interpolation.
    q: torch [views, rows, cols]. With fdkWeighting the FDK distance weight (SOD / U)^2 is applied.
    `coverage` (torch tensor of the grid shape) receives the summed angular interval of the views
    in which each voxel projects inside the detector. Returns torch [nz, ny, nx]."""
    import torch
    import torch.nn.functional as F
    device = device or q.device
    views = list(range(geometry.nViews)) if views is None else list(views)
    xs, ys, zs = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in grid.axes())
    volume = torch.zeros(grid.shape, dtype=torch.float32, device=device)
    rows, cols = geometry.rows, geometry.columns
    halfC, halfR = 0.5 * (cols - 1), 0.5 * (rows - 1)
    chunks = _sliceChunks(grid, maxVoxels)
    total = len(views) * len(chunks)
    done = 0
    for z0, z1 in chunks:
        z = zs[z0:z1][:, None, None]
        y = ys[None, :, None]
        x = xs[None, None, :]
        for i in views:
            if done % 8 == 0:
                _check(progress, shouldAbort, label, done, total,
                       f"{label}: view {views.index(i) + 1}/{len(views)}"
                       + (f", slices {z0 + 1}-{z1} of {grid.nz}" if len(chunks) > 1 else ""))
            done += 1
            s = geometry.source[i]
            n_, eu, ev, c = geometry.normal[i], geometry.eu[i], geometry.ev[i], geometry.detCenter[i]
            # Depth along the detector normal and position on the detector plane
            depth = (x - s[0]) * n_[0] + (y - s[1]) * n_[1] + (z - s[2]) * n_[2]
            scale = float(geometry.sdd[i]) / depth
            # hit = s + (P - s) * scale; detector coordinate = (hit - c) . e
            su = float(np.dot(s - c, eu))
            sv = float(np.dot(s - c, ev))
            pu = (x - s[0]) * eu[0] + (y - s[1]) * eu[1] + (z - s[2]) * eu[2]
            pv = (x - s[0]) * ev[0] + (y - s[1]) * ev[1] + (z - s[2]) * ev[2]
            uMm = su + pu * scale
            vMm = sv + pv * scale
            gx = (uMm / geometry.du) / halfC if halfC > 0 else uMm * 0
            gy = (vMm / geometry.dv) / halfR if halfR > 0 else vMm * 0
            grid2d = torch.stack(torch.broadcast_tensors(gx, gy), dim=-1).reshape(1, 1, -1, 2)
            sample = F.grid_sample(q[i][None, None], grid2d, mode="bilinear", padding_mode="zeros",
                                   align_corners=True).reshape(z1 - z0, grid.ny, grid.nx)
            if fdkWeighting:
                weight = (float(geometry.sod[i]) / depth) ** 2
                sample = sample * weight
            volume[z0:z1] += sample
            if coverage is not None:
                inside = (gx.abs() <= 1.0) & (gy.abs() <= 1.0)
                coverage[z0:z1] += inside.to(torch.float32).expand(z1 - z0, grid.ny, grid.nx) * float(geometry.dBeta[i])
    return volume


def fdk(p, geometry, grid, filterName=DEFAULT_FILTER, cutoff=1.0, truncationExtension=0.25,
        device=None, progress=None, shouldAbort=None, maskOutsideFov=True, log=print, cutoffV=None):
    """FDK reconstruction of line integrals p (numpy [views, rows, cols]). Returns numpy [nz, ny, nx]
    in 1/mm. cutoffV: optional Hann cut-off along the detector's vertical axis."""
    import torch
    device = device or torchDevice()
    shortScan = geometry.shortScanRange()
    if geometry.scanRange < math.pi:
        log(f"WARNING: the run covers only {math.degrees(geometry.scanRange):.0f} deg - limited-angle data, "
            "expect strong streaks.")
    elif geometry.scanRange < shortScan - math.radians(1.0):
        log(f"Note: the run covers {math.degrees(geometry.scanRange):.1f} deg, slightly less than the "
            f"{math.degrees(shortScan):.1f} deg a short scan needs; the outer part of the fan is weighted "
            "approximately.")
    q, filtered = filterProjections(p, geometry, filterName, cutoff, truncationExtension, device=device,
                                    progress=progress, shouldAbort=shouldAbort, cutoffV=cutoffV)
    padded = filtered is not geometry
    # Coverage of the padded detector is not the field of view: fovMask then does a separate pass
    coverage = torch.zeros(grid.shape, dtype=torch.float32, device=device) if maskOutsideFov and not padded else None
    volume = backproject(q, filtered, grid, fdkWeighting=True, device=device, progress=progress,
                         shouldAbort=shouldAbort, coverage=coverage)
    del q
    if coverage is not None:
        volume[coverage < FOV_COVERAGE * float(geometry.dBeta.sum())] = 0.0
    elif maskOutsideFov:
        mask = fovMask(geometry, grid, device=device, progress=progress, shouldAbort=shouldAbort)
        volume[torch.as_tensor(~mask, device=volume.device)] = 0.0
    return volume.cpu().numpy()


def fovMask(geometry, grid, device=None, progress=None, shouldAbort=None):
    """Boolean numpy [nz, ny, nx]: the region FDK can reconstruct. Centred detector: voxels seen by
    (nearly) every view. Displaced detector over a full rotation: voxels seen in at least half of the
    rotation and within the reach of the wide side of the detector."""
    import torch
    device = device or torchDevice()
    coverage = torch.zeros(grid.shape, dtype=torch.float32, device=device)
    dummy = torch.zeros((geometry.nViews, 1, 1), dtype=torch.float32, device=device)  # only coverage is used
    backproject(dummy, geometry, grid, fdkWeighting=False, device=device, progress=progress,
                shouldAbort=shouldAbort, label="Field of view", coverage=coverage)
    coverage = (coverage / float(geometry.dBeta.sum())).cpu().numpy()
    if not (isDisplacedDetector(geometry) and isFullRotation(geometry)):
        return coverage >= FOV_COVERAGE
    halfWidth = 0.5 * geometry.columns * geometry.du
    reach = float(np.min(geometry.sod * np.sin(np.arctan((halfWidth + np.abs(geometry.u0)) / geometry.sdd))))
    # Rotation axis: normal of the plane of the source trajectory, through the centre of the circle
    # fitted to it (the mean of the sources is off when the views are not spread evenly)
    mean = geometry.source.mean(axis=0)
    axis = np.linalg.svd(geometry.source - mean)[2][-1]
    a = np.hstack([2.0 * geometry.source, np.ones((geometry.nViews, 1))])
    a = np.vstack([a, np.append(axis, 0.0)])                    # keep the centre in the source plane:
    b = np.append(np.sum(geometry.source ** 2, axis=1), float(axis @ mean))   # axis . c = axis . mean
    centre = np.linalg.lstsq(a, b, rcond=None)[0][:3]
    xs, ys, zs = grid.axes()
    x = (xs - centre[0]).astype(np.float32)[None, None, :]
    y = (ys - centre[1]).astype(np.float32)[None, :, None]
    z = (zs - centre[2]).astype(np.float32)[:, None, None]
    along = x * axis[0] + y * axis[1] + z * axis[2]
    radial2 = x * x + y * y + z * z - along * along
    return (coverage >= 0.5) & (radial2 <= reach * reach)


# ---------------------------------------------------------------------------
# Forward projection and iterative refinement
# ---------------------------------------------------------------------------
def _rayBoxIntersection(source, directions, lower, upper):
    """Entry/exit distances of rays (source + t * direction) through an axis-aligned box."""
    import torch
    inv = 1.0 / torch.where(directions.abs() < 1e-12, torch.full_like(directions, 1e-12), directions)
    t1 = (lower - source) * inv
    t2 = (upper - source) * inv
    tNear = torch.minimum(t1, t2).amax(dim=-1)
    tFar = torch.maximum(t1, t2).amin(dim=-1)
    tNear = torch.clamp(tNear, min=0.0)
    return tNear, torch.maximum(tFar, tNear)


def forwardProject(volume, geometry, grid, views=None, device=None, stepFraction=0.75,
                   progress=None, shouldAbort=None, label="Projecting", sampleChunk=48):
    """Ray-driven projection of a volume (torch [nz, ny, nx], 1/mm) -> torch [len(views), rows, cols]
    line integrals. Rays are sampled with trilinear interpolation every stepFraction voxels."""
    import torch
    import torch.nn.functional as F
    device = device or volume.device
    views = list(range(geometry.nViews)) if views is None else list(views)
    rows, cols = geometry.rows, geometry.columns
    lower = torch.as_tensor(grid.origin - 0.5 * np.array([grid.sx, grid.sy, grid.sz]), dtype=torch.float32, device=device)
    upper = torch.as_tensor(grid.origin + grid.extent() + 0.5 * np.array([grid.sx, grid.sy, grid.sz]),
                            dtype=torch.float32, device=device)
    origin = torch.as_tensor(grid.origin, dtype=torch.float32, device=device)
    extent = torch.as_tensor(np.maximum(grid.extent(), 1e-6), dtype=torch.float32, device=device)
    step = stepFraction * min(grid.sx, grid.sy, grid.sz)
    u = torch.as_tensor(geometry.columnPositions(), dtype=torch.float32, device=device)
    v = torch.as_tensor(geometry.rowPositions(), dtype=torch.float32, device=device)
    vol5 = volume[None, None]
    out = torch.zeros((len(views), rows, cols), dtype=torch.float32, device=device)
    for k, i in enumerate(views):
        if k % 4 == 0:
            _check(progress, shouldAbort, label, k, len(views), f"{label}: view {k + 1}/{len(views)}")
        s = torch.as_tensor(geometry.source[i], dtype=torch.float32, device=device)
        c = torch.as_tensor(geometry.detCenter[i], dtype=torch.float32, device=device)
        eu = torch.as_tensor(geometry.eu[i], dtype=torch.float32, device=device)
        ev = torch.as_tensor(geometry.ev[i], dtype=torch.float32, device=device)
        pixels = c + u[None, :, None] * eu + v[:, None, None] * ev   # [rows, cols, 3]
        direction = pixels - s
        direction = direction / direction.norm(dim=-1, keepdim=True)
        tNear, tFar = _rayBoxIntersection(s, direction, lower, upper)
        length = tFar - tNear
        nSamples = int(math.ceil(float(length.max()) / step)) if float(length.max()) > 0 else 0
        total = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        for k0 in range(0, nSamples, sampleChunk):
            k1 = min(nSamples, k0 + sampleChunk)
            t = tNear[None] + (torch.arange(k0, k1, device=device, dtype=torch.float32)[:, None, None] + 0.5) * step
            valid = (t < tFar[None]).to(torch.float32)
            points = s + direction[None] * t[..., None]          # [K, rows, cols, 3]
            normalized = (points - origin) / extent * 2.0 - 1.0  # x, y, z order matches grid_sample
            samples = F.grid_sample(vol5, normalized[None], mode="bilinear", padding_mode="zeros",
                                    align_corners=True)[0, 0]
            total += (samples * valid).sum(dim=0)
        out[k] = total * step
    return out


def _gaussianBlur2d(images, sigma):
    """Separable Gaussian blur of torch [n, rows, cols] (sigma in pixels), edges replicated."""
    import torch
    import torch.nn.functional as F
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    out = images[:, None]
    out = F.conv2d(F.pad(out, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    out = F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))
    return out[:, 0]


def removeLowFrequencyInconsistency(measured, volume, geometry, grid, sigmaMm=10.0, batch=8,
                                    progress=None, shouldAbort=None):
    """Subtract from measured (torch [views, rows, cols], in place) the smooth part of
    measured - forwardProject(volume). Measured line integrals carry low-frequency terms no volume in
    the grid explains (an I0 or air level off zero, mass beyond the grid, scatter); FDK's ramp filter
    all but ignores them, SART spreads them over the grid as rings and fog. Removing them keeps the
    FDK's scale and lets SART refine the detail only. sigmaMm: Gaussian width at the detector.
    Returns the mean removed offset."""
    sigma = sigmaMm / geometry.du
    total = 0.0
    for start in range(0, geometry.nViews, batch):
        views = list(range(start, min(geometry.nViews, start + batch)))
        _check(progress, shouldAbort, "SART", 0, 1, f"Data consistency: views {start + 1}-{views[-1] + 1}")
        estimate = forwardProject(volume, geometry, grid, views)
        bias = _gaussianBlur2d(measured[views] - estimate, sigma)
        measured[views] -= bias
        total += float(bias.sum())
    return total / measured.numel()


def sart(p, geometry, grid, iterations=3, subsets=12, relaxation=0.5, initial=None, device=None,
         progress=None, shouldAbort=None, log=print, nonNegative=True, consistencySigmaMm=10.0):
    """Ordered-subsets SART. p: numpy line integrals [views, rows, cols]; initial: numpy volume
    (e.g. the FDK result) or None. With an initial volume and consistencySigmaMm, the low-frequency
    mismatch between the data and the initial volume is removed first (see
    removeLowFrequencyInconsistency). Returns numpy [nz, ny, nx] in 1/mm."""
    import torch
    device = device or torchDevice()
    measured = torch.as_tensor(p, dtype=torch.float32, device=device)
    volume = (torch.as_tensor(initial, dtype=torch.float32, device=device).clone() if initial is not None
              else torch.zeros(grid.shape, dtype=torch.float32, device=device))
    if initial is not None and consistencySigmaMm:
        offset = removeLowFrequencyInconsistency(measured, volume, geometry, grid, consistencySigmaMm,
                                                 progress=progress, shouldAbort=shouldAbort)
        log(f"Removed the low-frequency data mismatch to the initial volume (mean {offset:.3f}).")
    ones = torch.ones(grid.shape, dtype=torch.float32, device=device)
    nViews = geometry.nViews
    subsets = max(1, min(subsets, nViews))
    # Interleaved subsets spread each subset over the whole arc
    subsetViews = [list(range(s, nViews, subsets)) for s in range(subsets)]
    rowSums = {}
    full = torch.zeros(measured.shape, dtype=torch.float32, device=device)  # residuals of one subset
    for it in range(iterations):
        for sIndex, views in enumerate(subsetViews):
            label = f"SART iteration {it + 1}/{iterations}, subset {sIndex + 1}/{subsets}"
            _check(progress, shouldAbort, "SART", it * subsets + sIndex, iterations * subsets, label)
            if sIndex not in rowSums:
                rowSums[sIndex] = forwardProject(ones, geometry, grid, views, device=device)
            estimate = forwardProject(volume, geometry, grid, views, device=device)
            residual = (measured[views] - estimate) / torch.clamp(rowSums[sIndex], min=1e-3)
            residual = torch.where(rowSums[sIndex] > 1e-3, residual, torch.zeros_like(residual))
            # Backproject without FDK weighting; normalise by how many views saw each voxel
            full[views] = residual
            count = torch.zeros(grid.shape, dtype=torch.float32, device=device)
            saved = geometry.dBeta
            geometry.dBeta = np.ones_like(saved)  # coverage then counts views
            try:
                update = backproject(full, geometry, grid, fdkWeighting=False, views=views, device=device,
                                     coverage=count)
            finally:
                geometry.dBeta = saved
                full[views] = 0.0
            volume += relaxation * update / torch.clamp(count, min=1.0)
            if nonNegative:
                volume.clamp_(min=0.0)
        log(f"SART iteration {it + 1}/{iterations} done.")
    return volume.cpu().numpy()


def estimateWaterAttenuation(volume, tissueHu=40.0):
    """Water attenuation (same units as volume) from the soft-tissue peak of the histogram, for an
    automatic Hounsfield scale: the most frequent value among voxels above 35% of the 99th
    percentile (which leaves out air, lung and the zeroed outside of the field of view) in the
    central part of the volume is taken as soft tissue at tissueHu."""
    nz, ny, nx = volume.shape
    z0, z1 = int(nz * 0.2), max(int(nz * 0.8), int(nz * 0.2) + 1)
    y, x = np.ogrid[:ny, :nx]
    inner = ((y - (ny - 1) / 2.0) / (0.3 * ny)) ** 2 + ((x - (nx - 1) / 2.0) / (0.3 * nx)) ** 2 <= 1.0
    values = volume[z0:z1][:, inner]
    values = values[values > 0]
    if values.size < 100:
        return None
    high = float(np.percentile(values, 99.0))
    values = values[values > 0.35 * high]
    if values.size < 100:
        return None
    counts, edges = np.histogram(values, bins=200, range=(0.35 * high, high))
    kernel = np.ones(7) / 7.0
    smooth = np.convolve(counts, kernel, mode="same")
    peak = int(np.argmax(smooth))
    tissue = 0.5 * (edges[peak] + edges[peak + 1])
    return float(tissue / (1.0 + tissueHu / 1000.0))


def toOutputUnits(volume, units="1/cm", muWaterPerCm=0.2):
    """1/mm -> 1/cm, or to Hounsfield-like units with the given water attenuation."""
    perCm = volume * 10.0
    if units == "HU":
        return (1000.0 * (perCm - muWaterPerCm) / muWaterPerCm).astype(np.float32)
    return perCm.astype(np.float32)

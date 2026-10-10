"""Cone-beam reconstruction for C-arm rotational runs (PyTorch, no Slicer).

    lineIntegrals   detector intensities -> line integrals  p = ln(I0 / I)
    fdk             Feldkamp-Davis-Kress filtered backprojection with Parker short-scan weights,
                    cosine pre-weighting, apodised ramp filter and lateral truncation correction
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
    """Lateral truncation correction: extend each detector row on both sides with its edge value
    fading to zero along a cosine (removes the bright rim of truncated projections)."""
    import torch
    if extension <= 0:
        return p
    ramp = 0.5 * (1.0 + torch.cos(torch.linspace(0, math.pi, extension, device=p.device, dtype=p.dtype)))
    left = p[..., :1] * ramp.flip(0)
    right = p[..., -1:] * ramp
    return torch.cat([left, p, right], dim=-1)


def filterProjections(p, geometry, filterName=DEFAULT_FILTER, cutoff=1.0, truncationExtension=0.25,
                      weights=None, device=None, progress=None, shouldAbort=None, batch=16):
    """Cosine pre-weighting, redundancy weights and ramp filtering along the detector rows.
    p: numpy [views, rows, cols] line integrals. Returns a torch tensor on `device`, already
    multiplied by the angular interval of each view, ready for backprojection."""
    import torch
    device = device or torchDevice()
    nViews, rows, cols = p.shape
    extension = int(round(truncationExtension * cols)) if truncationExtension else 0
    nTotal = cols + 2 * extension
    nPad = 1 << int(math.ceil(math.log2(2 * nTotal)))
    tauDetector = geometry.du
    u = torch.as_tensor(geometry.columnPositions(), dtype=torch.float32, device=device)
    v = torch.as_tensor(geometry.rowPositions(), dtype=torch.float32, device=device)
    if weights is None:
        weights = parkerWeights(geometry)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    out = torch.empty((nViews, rows, cols), dtype=torch.float32, device=device)
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
            row = block[k] * cosine
            row = _extendRows(row, extension)
            # Filter on the virtual detector through the isocenter (pixel size du * SOD / SDD)
            tau = tauDetector * sod / sdd
            key = round(tau, 6)
            if key not in kernels:
                kernels[key] = torch.as_tensor(rampKernel(nPad, tau, filterName, cutoff), device=device)
            spectrum = torch.fft.rfft(row, n=nPad, dim=-1) * kernels[key]
            filtered = torch.fft.irfft(spectrum, n=nPad, dim=-1)[:, extension:extension + cols]
            out[i] = filtered * weights[i][None, :] * float(geometry.dBeta[i])
    return out


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
        device=None, progress=None, shouldAbort=None, maskOutsideFov=True, log=print):
    """FDK reconstruction of line integrals p (numpy [views, rows, cols]). Returns numpy [nz, ny, nx]
    in 1/mm."""
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
    q = filterProjections(p, geometry, filterName, cutoff, truncationExtension, device=device,
                          progress=progress, shouldAbort=shouldAbort)
    coverage = torch.zeros(grid.shape, dtype=torch.float32, device=device) if maskOutsideFov else None
    volume = backproject(q, geometry, grid, fdkWeighting=True, device=device, progress=progress,
                         shouldAbort=shouldAbort, coverage=coverage)
    if coverage is not None:
        volume[coverage < FOV_COVERAGE * float(geometry.dBeta.sum())] = 0.0
    del q
    return volume.cpu().numpy()


def fovMask(geometry, grid, device=None, progress=None, shouldAbort=None):
    """Boolean numpy [nz, ny, nx]: voxels seen by (nearly) every view, the region FDK keeps."""
    import torch
    device = device or torchDevice()
    coverage = torch.zeros(grid.shape, dtype=torch.float32, device=device)
    dummy = torch.zeros((geometry.nViews, 1, 1), dtype=torch.float32, device=device)  # only coverage is used
    backproject(dummy, geometry, grid, fdkWeighting=False, device=device, progress=progress,
                shouldAbort=shouldAbort, label="Field of view", coverage=coverage)
    return (coverage >= FOV_COVERAGE * float(geometry.dBeta.sum())).cpu().numpy()


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


def sart(p, geometry, grid, iterations=3, subsets=12, relaxation=0.5, initial=None, device=None,
         progress=None, shouldAbort=None, log=print, nonNegative=True):
    """Ordered-subsets SART. p: numpy line integrals [views, rows, cols]; initial: numpy volume
    (e.g. the FDK result) or None. Returns numpy [nz, ny, nx] in 1/mm."""
    import torch
    device = device or torchDevice()
    measured = torch.as_tensor(p, dtype=torch.float32, device=device)
    volume = (torch.as_tensor(initial, dtype=torch.float32, device=device).clone() if initial is not None
              else torch.zeros(grid.shape, dtype=torch.float32, device=device))
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

"""Tests of the cone-beam reconstruction on synthetic data (no Slicer needed).

Run from the module folder:  python -m unittest discover -s Testing/Python -v
Needs numpy, torch and pydicom (SimpleITK is optional).
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from DICOMCBCTReconLib import gecalib  # noqa: E402
from DICOMCBCTReconLib import geometry as G  # noqa: E402
from DICOMCBCTReconLib import phantom as P  # noqa: E402
from DICOMCBCTReconLib import reconstruction as R  # noqa: E402
from DICOMCBCTReconLib import xadata  # noqa: E402

GRID = G.ReconGrid((96, 96, 48), (1.8, 1.8, 1.8))
CENTRAL = (slice(16, 32), slice(22, 74), slice(22, 74))
SILENT = lambda *args: None  # noqa: E731


def centralError(volume, truth):
    return float(np.abs(volume - truth)[CENTRAL].mean())


class GeometryTest(unittest.TestCase):
    def test_ap_view_conventions(self):
        # At primary = secondary = 0 the tube is posterior and the detector anterior,
        # image columns run to the patient's left and rows to the feet
        header = P.syntheticRunHeader(nFrames=2, startDeg=0.0, arcDeg=1.0)
        geo = G.ProjectionGeometry(header)
        np.testing.assert_allclose(geo.source[0], [0.0, 720.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(geo.detCenter[0], [0.0, -473.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(geo.eu[0], [1.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(geo.ev[0], [0.0, 0.0, -1.0], atol=1e-9)
        column, row = geo.projectPoints(np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]), 0)
        self.assertGreater(column[0], 0.5 * (geo.columns - 1))   # patient's left -> right of image
        self.assertLess(row[1], 0.5 * (geo.rows - 1))             # head -> top of image

    def test_lao_moves_detector_left(self):
        header = P.syntheticRunHeader(nFrames=2, startDeg=90.0, arcDeg=1.0)
        geo = G.ProjectionGeometry(header)
        self.assertGreater(geo.detCenter[0][0], 400.0)  # LAO 90: detector on the patient's left

    def test_cumulative_and_step_increments(self):
        cumulative, how = xadata.anglesFromIncrements(-100.0, [0, 0.5, 1.0, 1.5] * 1 + list(np.arange(2, 50, 0.5)))
        self.assertAlmostEqual(cumulative[-1], -100.0 + 49.5)
        steps, how = xadata.anglesFromIncrements(-100.0, [0.5] * 100, log=SILENT)
        self.assertAlmostEqual(steps[-1], -100.0 + 49.5)
        self.assertEqual(how, "per-frame steps")


class SweepAndEncodingTest(unittest.TestCase):
    def test_two_sweeps_are_split(self):
        # GE subtraction spin: increments restart at the first frame of the second sweep
        increments = list(np.round(np.linspace(0, 192.6, 147), 1)) * 2
        self.assertEqual(xadata.findSweeps(increments), [(0, 147), (147, 294)])
        self.assertEqual(xadata.findSweeps(increments[:147]), [(0, 147)])

    def test_sqrt_encoding_doubles_line_integrals(self):
        p = np.full((1, 4, 4), 1.5, dtype=np.float32)
        linear = 4000.0 * np.exp(-p)
        encoded = np.sqrt(linear / 4000.0) * 4000.0
        np.testing.assert_allclose(R.lineIntegrals(linear, R.INTENSITY_LOG, i0=4000.0), 1.5, rtol=1e-5)
        np.testing.assert_allclose(R.lineIntegrals(encoded, R.INTENSITY_SQRT, i0=4000.0), 1.5, rtol=1e-5)
        self.assertEqual(R.defaultIntensityMode({"intensityRelationship": "SQRT"}), R.INTENSITY_SQRT)

    def test_water_estimate_from_soft_tissue_peak(self):
        grid = G.ReconGrid((64, 64, 32), (3.0, 3.0, 3.0))
        volume = P.voxelizePhantom(grid)
        water = R.estimateWaterAttenuation(volume)
        self.assertAlmostEqual(water * 1.04, 0.020, delta=0.001)  # body = soft tissue at 40 HU


class CalibrationTest(unittest.TestCase):
    def test_matrix_round_trip(self):
        geo = G.ProjectionGeometry(P.syntheticRunHeader(nFrames=20, arcDeg=200.0))
        for i in (0, 9, 19):
            M = G.matrixFromVectors(geo.source[i], geo.detCenter[i], geo.eu[i], geo.ev[i],
                                    geo.du, geo.dv, geo.rows, geo.columns)
            source, center, eu, ev, dv = G.vectorsFromMatrix(M * 3.7, geo.du, geo.rows, geo.columns)
            np.testing.assert_allclose(source, geo.source[i], atol=1e-6)
            np.testing.assert_allclose(center, geo.detCenter[i], atol=1e-6)
            np.testing.assert_allclose(eu, geo.eu[i], atol=1e-9)
            np.testing.assert_allclose(ev, geo.ev[i], atol=1e-9)
            self.assertAlmostEqual(dv, geo.dv)

    def test_calibration_recovers_real_trajectory(self):
        """A C-arm whose detector shifts and source sags with angle: reconstructing with the
        (synthetic) GE calibration must be as good as with the true geometry, and better than nominal."""
        header = P.syntheticRunHeader(nFrames=147, arcDeg=192.6, rows=96, columns=96, pixelMm=3.2, startDeg=-102.4)
        perturbation = {"shiftU": 3.0, "shiftV": 2.0, "sag": 2.5}
        true = P.perturbGeometry(G.ProjectionGeometry(header), **perturbation)
        p = P.projectPhantom(true)
        calibration = P.syntheticCalibration(header, np.linspace(-104.3, 92.7, 152), perturbation)
        calibrated = G.ProjectionGeometry(dict(header, calibration=calibration))
        self.assertTrue(calibrated.calibrated)
        self.assertLess(calibrated.calibrationInfo["rms"], 1.0)
        self.assertAlmostEqual(calibrated.calibrationInfo["scale"], 0.1, delta=1e-4)
        self.assertFalse(calibrated.calibrationInfo["reflected"])
        truth = P.voxelizePhantom(GRID)
        errors = {name: centralError(R.fdk(p, g, GRID, truncationExtension=0, log=SILENT), truth)
                  for name, g in (("true", true), ("calibrated", calibrated),
                                  ("nominal", G.ProjectionGeometry(header)))}
        self.assertLess(errors["calibrated"], errors["true"] * 1.02)
        self.assertLess(errors["calibrated"] * 1.15, errors["nominal"])

    def test_mismatched_calibration_is_refused(self):
        header = P.syntheticRunHeader(nFrames=60, arcDeg=192.6, startDeg=-102.4)
        calibration = P.syntheticCalibration(header, np.linspace(-104.3, 92.7, 152))
        wrong = dict(header, sod=np.full(60, 400.0), calibration=calibration)
        matrices, info = gecalib.frameMatrices(wrong)
        self.assertIsNone(matrices)
        self.assertIn("reason", info)


class ParkerTest(unittest.TestCase):
    def test_conjugate_rays_sum_to_one(self):
        """Within the central plane, each ray and its opposite ray must get weights summing to 1."""
        header = P.syntheticRunHeader(nFrames=721, arcDeg=200.0)
        geo = G.ProjectionGeometry(header)
        weights = R.parkerWeights(geo)
        u = geo.columnPositions()
        rng = np.random.default_rng(0)
        checked = 0
        for _ in range(300):
            i = int(rng.integers(0, geo.nViews))
            j = int(rng.integers(geo.columns // 8, geo.columns - geo.columns // 8))
            pixel = geo.detCenter[i] + u[j] * geo.eu[i]
            # Opposite ray: the view whose source lies on the line through this ray
            direction = pixel - geo.source[i]
            direction /= np.linalg.norm(direction)
            rel = geo.source - geo.source[i]
            distance = np.linalg.norm(rel - np.outer(rel @ direction, direction), axis=1)
            distance[max(0, i - 5):i + 6] = np.inf
            k = int(np.argmin(distance))
            if distance[k] > 2.0:
                continue  # conjugate outside the scanned arc: weight must then be ~1
            columnK, _ = geo.projectPoints(geo.source[i][None], k)
            wk = np.interp(columnK[0], np.arange(geo.columns), weights[k])
            self.assertAlmostEqual(weights[i, j] + wk, 1.0, delta=0.06)
            checked += 1
        self.assertGreater(checked, 50)


class FdkTest(unittest.TestCase):
    def _reconstruct(self, arcDeg, nFrames, **kwargs):
        header = P.syntheticRunHeader(nFrames=nFrames, arcDeg=arcDeg, rows=96, columns=96, pixelMm=3.2)
        geo = G.ProjectionGeometry(header)
        p = P.projectPhantom(geo)
        return R.fdk(p, geo, GRID, log=SILENT, **kwargs), P.voxelizePhantom(GRID)

    def test_short_scan_accuracy_both_directions(self):
        for arc in (200.0, -200.0):
            volume, truth = self._reconstruct(arc, 151, truncationExtension=0)
            self.assertAlmostEqual(float(volume[24, 48, 48]), 0.020, delta=0.0015)
            self.assertLess(centralError(volume, truth), 0.0016, f"arc {arc}")

    def test_wrong_parker_direction_is_worse(self):
        header = P.syntheticRunHeader(nFrames=151, arcDeg=200.0, rows=96, columns=96, pixelMm=3.2)
        geo = G.ProjectionGeometry(header)
        p = P.projectPhantom(geo)
        truth = P.voxelizePhantom(GRID)
        good = centralError(R.fdk(p, geo, GRID, truncationExtension=0, log=SILENT), truth)
        geo.rotationSign = -geo.rotationSign
        bad = centralError(R.fdk(p, geo, GRID, truncationExtension=0, log=SILENT), truth)
        self.assertLess(good * 1.4, bad)

    def test_truncation_correction_reduces_cupping(self):
        wide = [((0.0, 0.0, 0.0), (150.0, 110.0, 90.0), 0.02)] + P.DEFAULT_PHANTOM[1:]
        header = P.syntheticRunHeader(nFrames=151, arcDeg=200.0, rows=96, columns=96, pixelMm=3.2)
        geo = G.ProjectionGeometry(header)
        p = P.projectPhantom(geo, wide)
        truth = P.voxelizePhantom(GRID, wide)
        plain = centralError(R.fdk(p, geo, GRID, truncationExtension=0, log=SILENT), truth)
        corrected = centralError(R.fdk(p, geo, GRID, truncationExtension=0.5, log=SILENT), truth)
        self.assertLess(corrected, plain)

    def test_uniform_cylinder_is_flat(self):
        # Centred detector over a short scan, and a detector displaced by 90 mm over a full rotation
        # (half fan: only the displaced-detector weights and padding reconstruct the wide cylinder)
        grid = G.ReconGrid((128, 128, 16), (1.8, 1.8, 1.8))
        x, y, _ = grid.axes()
        r = np.hypot(*np.meshgrid(x, y, indexing="xy"))
        for arc, nFrames, shift, radius in ((200.0, 151, 0.0, 70.0), (360.0, 240, 90.0, 110.0)):
            header = P.syntheticRunHeader(nFrames=nFrames, arcDeg=arc, rows=96, columns=96, pixelMm=3.2)
            geo = G.ProjectionGeometry(header, {"offsetU": shift})
            self.assertEqual(R.isDisplacedDetector(geo), shift != 0.0)
            cylinder = [((0.0, 0.0, 0.0), (radius, radius, 400.0), 0.02)]
            volume = R.fdk(P.projectPhantom(geo, cylinder), geo, grid, "Hann", 0.7, log=SILENT, cutoffV=0.7)
            middle = volume[6:10]
            rings = [float(middle[:, (r >= a) & (r < a + 10.0)].mean()) for a in np.arange(0.0, 0.8 * radius, 10.0)]
            self.assertLess(max(abs(m - 0.02) for m in rings), 0.0003, f"arc {arc}, shift {shift}: {rings}")


class ProjectorAndSartTest(unittest.TestCase):
    def test_forward_projector_matches_analytic(self):
        import torch
        header = P.syntheticRunHeader(nFrames=12, arcDeg=200.0, rows=48, columns=48, pixelMm=6.4)
        geo = G.ProjectionGeometry(header)
        grid = G.ReconGrid((128, 128, 96), (1.5, 1.5, 1.5))
        smooth = [((0.0, 0.0, 0.0), (60.0, 50.0, 50.0), 0.02)]
        volume = torch.as_tensor(P.voxelizePhantom(grid, smooth))
        projected = R.forwardProject(volume, geo, grid, device=torch.device("cpu")).numpy()
        exact = P.projectPhantom(geo, smooth)
        inner = exact > 0.5
        relative = np.abs(projected[inner] - exact[inner]) / exact[inner]
        self.assertLess(float(np.median(relative)), 0.02)

    def test_sart_does_not_degrade_fdk(self):
        header = P.syntheticRunHeader(nFrames=60, arcDeg=200.0, rows=64, columns=64, pixelMm=4.8)
        geo = G.ProjectionGeometry(header)
        grid = G.ReconGrid((64, 64, 32), (2.7, 2.7, 2.7))
        p = P.projectPhantom(geo)
        truth = P.voxelizePhantom(grid)
        start = R.fdk(p, geo, grid, truncationExtension=0, log=SILENT)
        refined = R.sart(p, geo, grid, iterations=2, subsets=6, relaxation=0.5, initial=start, log=SILENT)
        core = (slice(10, 22), slice(12, 52), slice(12, 52))
        before = float(np.abs(start - truth)[core].mean())
        after = float(np.abs(refined - truth)[core].mean())
        self.assertLess(after, before * 1.05)

    def test_sart_ignores_smooth_data_offset(self):
        # An offset on every line integral (I0 off, air not at zero) is mostly filtered out by FDK;
        # SART starting from FDK must keep the FDK level rather than turn the offset into fog
        header = P.syntheticRunHeader(nFrames=60, arcDeg=200.0, rows=64, columns=64, pixelMm=4.8)
        geo = G.ProjectionGeometry(header)
        grid = G.ReconGrid((64, 64, 16), (2.7, 2.7, 2.7))
        cylinder = [((0.0, 0.0, 0.0), (60.0, 60.0, 400.0), 0.02)]
        p = P.projectPhantom(geo, cylinder) + 0.5
        start = R.fdk(p, geo, grid, log=SILENT)
        refined = R.sart(p, geo, grid, iterations=2, subsets=6, initial=start, log=SILENT)
        plain = R.sart(p, geo, grid, iterations=2, subsets=6, initial=start, log=SILENT, consistencySigmaMm=0)
        core = (slice(6, 10), slice(18, 46), slice(18, 46))
        level = float(start[core].mean())
        self.assertAlmostEqual(float(refined[core].mean()), level, delta=0.0001)
        self.assertGreater(float(plain[core].mean()), level + 0.0004)


class DicomRoundTripTest(unittest.TestCase):
    def test_synthetic_run(self):
        header = P.syntheticRunHeader(nFrames=121, arcDeg=200.0, rows=96, columns=96, pixelMm=3.2)
        geo = G.ProjectionGeometry(header)
        p = P.projectPhantom(geo)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "run.dcm")
            P.writeSyntheticRun(path, header, p, i0=4000.0)
            ds = xadata.readHeader(path)
            ok, reason = xadata.isRotationalRun(ds)
            self.assertTrue(ok, reason)
            read = xadata.readGeometry(ds, log=SILENT)
            self.assertIn("GE per-frame", read["distanceSource"])
            np.testing.assert_allclose(read["primaryDeg"], header["primaryDeg"], atol=1e-3)
            frames = xadata.readFrames(path, read, log=SILENT)
        self.assertEqual(frames.shape, (121, 96, 96))
        integrals = R.lineIntegrals(frames, R.INTENSITY_LOG, i0=4000.0)
        volume = R.fdk(integrals, G.ProjectionGeometry(read), GRID, truncationExtension=0, log=SILENT)
        truth = P.voxelizePhantom(GRID)
        self.assertLess(centralError(volume, truth), 0.002)

    def test_subtraction_spin_with_calibration(self):
        """Two sweeps in one file, square-root encoded, with a GE calibration block."""
        n = 91
        header = P.syntheticRunHeader(nFrames=n, arcDeg=200.0, rows=96, columns=96, pixelMm=3.2)
        geo = G.ProjectionGeometry(header)
        mask = P.projectPhantom(geo)
        vessel = [((5.0, -20.0, -10.0), (4.0, 4.0, 35.0), 0.03)]
        contrast = mask + P.projectPhantom(geo, vessel)
        both = dict(header, nFrames=2 * n, primaryDeg=np.concatenate([header["primaryDeg"]] * 2),
                    secondaryDeg=np.zeros(2 * n), sid=np.full(2 * n, 1193.0), sod=np.full(2 * n, 720.0))
        calibration = P.syntheticCalibration(header, np.linspace(-104.0, 99.0, 120))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "spin.dcm")
            P.writeSyntheticRun(path, both, np.concatenate([mask, contrast]), encoding="SQRT",
                                calibration=calibration, subtractionSpin=True)
            ds = xadata.readHeader(path)
            self.assertTrue(xadata.isRotationalRun(ds)[0])
            run = xadata.readGeometry(ds, log=SILENT)
            self.assertEqual(run["sweeps"], [(0, n), (n, 2 * n)])
            self.assertTrue(run["subtractionSpin"])
            self.assertEqual(xadata.sweepLabels(run), ["sweep 1 (mask)", "sweep 2 (contrast)"])
            self.assertIsNotNone(run["calibration"])
            self.assertEqual(len(run["calibration"]["matrices"]), 120)
            frames = xadata.readFrames(path, run, log=SILENT)
        sweeps = [xadata.sweepGeometry(run, k) for k in (0, 1)]
        self.assertEqual(R.defaultIntensityMode(run), R.INTENSITY_SQRT)
        integrals = [R.lineIntegrals(frames[a:b], R.INTENSITY_SQRT, i0=4000.0) for a, b in run["sweeps"]]
        geometries = [G.ProjectionGeometry(s) for s in sweeps]
        self.assertTrue(all(g.calibrated for g in geometries))
        # Mask sweep alone reconstructs the anatomy
        anatomy = R.fdk(integrals[0], geometries[0], GRID, truncationExtension=0, log=SILENT)
        self.assertLess(centralError(anatomy, P.voxelizePhantom(GRID)), 0.002)
        # Contrast minus mask leaves the vessel only
        dsa = R.fdk(integrals[1] - integrals[0], geometries[1], GRID, truncationExtension=0, log=SILENT)
        vesselTruth = P.voxelizePhantom(GRID, vessel)
        inside, outside = dsa[vesselTruth > 0], dsa[vesselTruth == 0]
        self.assertGreater(float(np.median(inside)), 0.02)
        self.assertLess(float(np.abs(np.median(outside))), 0.001)

    def test_non_rotational_object_is_rejected(self):
        import pydicom
        ds = pydicom.Dataset()
        ds.Modality = "CT"
        self.assertFalse(xadata.isRotationalRun(ds)[0])


if __name__ == "__main__":
    unittest.main()

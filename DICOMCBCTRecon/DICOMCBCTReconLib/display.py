"""Preview rendering: numpy slices -> grey RGB arrays -> QPixmap, entirely in memory.

Taken from the SPECT reconstruction plugin (DICOMSpectReconPluginLib.display): slices are turned
into 8-bit RGB with numpy, wrapped in a binary PPM header and decoded by Qt's built-in PPM reader.
If that does not work in a given Slicer/PythonQt build, the renderer falls back once,
permanently, to temporary PNG files.
"""

import logging
import os
import shutil
import tempfile
import traceback

import numpy as np
import qt


def displayWindow(arr, low=0.5, high=99.8):
    """(minimum, maximum) display range from percentiles of the non-zero voxels/pixels."""
    if arr is None or not arr.size:
        return 0.0, 1.0
    values = arr[arr != 0]
    if not values.size:
        return 0.0, 1.0
    step = max(1, values.size // 2_000_000)
    lo, hi = np.percentile(values[::step], [low, high])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def grayscale(arr, window):
    """Scalar slice -> grey RGB uint8 [h, w, 3] with window = (minimum, maximum)."""
    lo, hi = window
    gray = (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(gray[..., np.newaxis], 3, axis=-1)


# ---------------------------------------------------------------------------
# RGB array -> QPixmap
# ---------------------------------------------------------------------------
def ppmBytes(rgb):
    """Binary PPM (P6) encoding of an RGB uint8 array whose row 0 is the top row."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[0], rgb.shape[1]
    return b"P6\n%d %d\n255\n" % (w, h) + rgb.tobytes()


class PixmapRenderer:
    """Turns RGB arrays into QPixmaps of a fixed display size."""

    def __init__(self, size):
        self.size = size
        self.useFiles = not self._inMemoryWorks()
        self.tempDir = None
        if self.useFiles:
            logging.warning("CBCT preview: in-memory image decoding unavailable, using temporary PNG files.")

    # --- in-memory path ---
    @staticmethod
    def _decode(rgb):
        return qt.QImage.fromData(ppmBytes(rgb), "PPM")

    def _inMemoryWorks(self):
        """Decode a 2x2 test image and check size, orientation and channel order."""
        try:
            test = np.array([[[255, 0, 0], [0, 255, 0]],
                             [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
            image = self._decode(test)
            if image is None or image.isNull() or image.width() != 2 or image.height() != 2:
                return False
            return (int(image.pixel(0, 0)) & 0xFFFFFF) == 0xFF0000 and \
                   (int(image.pixel(1, 1)) & 0xFFFFFF) == 0xFFFFFF and \
                   (int(image.pixel(0, 1)) & 0xFFFFFF) == 0x0000FF
        except Exception:
            logging.warning("CBCT preview: in-memory image test failed:\n" + traceback.format_exc())
            return False

    # --- fallback path (previous behaviour) ---
    def _viaFile(self, rgb):
        import vtk
        import vtk.util.numpy_support as numpy_support
        if self.tempDir is None:
            self.tempDir = tempfile.mkdtemp(prefix="cbctrecon_preview_")
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[0], rgb.shape[1]
        vtkImg = vtk.vtkImageData()
        vtkImg.SetDimensions(w, h, 1)
        # VTK images are bottom-up, flip so row 0 is shown at the top
        flipped = np.ascontiguousarray(rgb[::-1])
        scalars = numpy_support.numpy_to_vtk(flipped.reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtkImg.GetPointData().SetScalars(scalars)
        path = os.path.join(self.tempDir, "preview.png")
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputData(vtkImg)
        writer.Write()
        # QImage (not QPixmap(path)) to avoid Qt's pixmap cache returning a stale image
        return qt.QImage(path)

    def toPixmap(self, rgb):
        image = None
        if not self.useFiles:
            image = self._decode(rgb)
            if image is None or image.isNull():
                logging.warning("CBCT preview: in-memory decoding failed, switching to temporary PNG files.")
                self.useFiles = True
                image = None
        if image is None:
            image = self._viaFile(rgb)
        pixmap = qt.QPixmap.fromImage(image)
        return pixmap.scaled(self.size, self.size, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)

    def cleanup(self):
        if self.tempDir:
            shutil.rmtree(self.tempDir, ignore_errors=True)
            self.tempDir = None

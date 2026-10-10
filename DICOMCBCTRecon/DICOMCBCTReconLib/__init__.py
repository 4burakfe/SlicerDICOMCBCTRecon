"""Helper package of the DICOMCBCTRecon module.

    xadata          XA/XRF rotational run: geometry from DICOM (incl. GE private tags), frames (numpy/pydicom)
    geometry        C-arm projection geometry (nominal or from projection matrices), reconstruction grid
    gecalib         GE Innova 3D calibration (GEMS_XR3DCAL_01) -> per-frame matrices in patient coordinates
    reconstruction  line integrals, FDK with Parker weights, forward projector, OS-SART (PyTorch)
    phantom         analytic ellipsoid phantom and synthetic DICOM runs for tests (numpy/pydicom)
    display         numpy -> in-memory preview images (Qt)
    dialog          the reconstruction window (Slicer)
"""

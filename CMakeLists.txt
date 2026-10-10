cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DICOMCBCTRecon)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/4burakfe/DICOMCBCTRecon")
set(EXTENSION_CATEGORY "Reconstruction")
set(EXTENSION_CONTRIBUTORS "Burak Demir")
set(EXTENSION_DESCRIPTION "Cone-beam CT reconstruction of rotational C-arm X-ray runs (XA/XRF multi-frame DICOM) with FDK and SART, loaded through the DICOM browser.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/4burakfe/DICOMCBCTRecon/main/DICOMCBCTRecon/Resources/Icons/DICOMCBCTRecon.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/4burakfe/DICOMCBCTRecon/main/screenshot.jpg")
set(EXTENSION_DEPENDS PyTorch)

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DICOMCBCTRecon)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

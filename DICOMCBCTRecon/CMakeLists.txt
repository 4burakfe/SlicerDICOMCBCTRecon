#-----------------------------------------------------------------------------
set(MODULE_NAME DICOMCBCTRecon)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/xadata.py
  ${MODULE_NAME}Lib/geometry.py
  ${MODULE_NAME}Lib/gecalib.py
  ${MODULE_NAME}Lib/reconstruction.py
  ${MODULE_NAME}Lib/phantom.py
  ${MODULE_NAME}Lib/display.py
  ${MODULE_NAME}Lib/dialog.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  # Self test inside Slicer (synthetic phantom)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()

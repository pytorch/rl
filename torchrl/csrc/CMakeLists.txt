################################################################################
# _torchrl.so
################################################################################
if (BUILD_TORCHRL_PYTHON_EXTENSION)
  set(
    EXTENSION_SOURCES
    pybind.cpp
    )
  add_library(
    _torchrl
    SHARED
    ${EXTENSION_SOURCES}
    )

  set_target_properties(_torchrl PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(_torchrl PROPERTIES SUFFIX ".pyd")
  endif(MSVC)

  if (APPLE)
    # https://github.com/facebookarchive/caffe2/issues/854#issuecomment-364538485
    # https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
    set_target_properties(_torchrl PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif()

  target_include_directories(
    _torchrl
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${Python_INCLUDE_DIR}
    )

  if (APPLE)
    # See https://github.com/pytorch/pytorch/issues/38122
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
  else()
    # PyTorch dependency
    execute_process(
      COMMAND
      python -c
      "import os; import torch; print(os.path.dirname(torch.__file__), end='')"
      OUTPUT_VARIABLE
      TORCH_PATH
    )
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
    find_package(Torch REQUIRED)
    set(TORCH_PYTHON_LIBRARIES "${TORCH_PATH}/lib/libtorch_python.so")
  endif()

  if (WIN32)
    find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)
    set(ADDITIONAL_ITEMS Python3::Python)
  endif()

  target_link_libraries(_torchrl PUBLIC torch ${TORCH_PYTHON_LIBRARIES})

  install(
    TARGETS _torchrl
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .  # For Windows
    )
endif()

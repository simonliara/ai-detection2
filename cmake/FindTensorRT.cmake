cmake_minimum_required(VERSION 3.16)

include(FindPackageHandleStandardArgs)

set(TENSORRT_ROOT "" CACHE PATH "Root directory of TensorRT installation")

set(_TRT_HINT_ROOTS "")
if(TENSORRT_ROOT)
  list(APPEND _TRT_HINT_ROOTS "${TENSORRT_ROOT}")
endif()
if(DEFINED ENV{TENSORRT_ROOT} AND NOT "$ENV{TENSORRT_ROOT}" STREQUAL "")
  list(APPEND _TRT_HINT_ROOTS "$ENV{TENSORRT_ROOT}")
endif()
if(DEFINED ENV{TENSORRT_DIR} AND NOT "$ENV{TENSORRT_DIR}" STREQUAL "")
  list(APPEND _TRT_HINT_ROOTS "$ENV{TENSORRT_DIR}")
endif()

list(APPEND _TRT_HINT_ROOTS
  "/usr/src/tensorrt"
  "/usr/local/TensorRT"
  "/usr/local/tensorrt"
  "/usr/local"
  "/usr"
)

file(GLOB _TRT_GLOB_ROOTS "/usr/local/TensorRT-*")
list(APPEND _TRT_HINT_ROOTS ${_TRT_GLOB_ROOTS})

list(REMOVE_DUPLICATES _TRT_HINT_ROOTS)

set(_TRT_INCLUDE_SUFFIXES
  "include"
  "include/aarch64-linux-gnu"
  "include/x86_64-linux-gnu"
)

set(_TRT_LIB_SUFFIXES
  "lib"
  "lib64"
  "lib/aarch64-linux-gnu"
  "lib/x86_64-linux-gnu"
)

find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${_TRT_HINT_ROOTS}
  PATH_SUFFIXES ${_TRT_INCLUDE_SUFFIXES}
)

function(_tensorrt_find_lib outvar libname)
  find_library(${outvar}
    NAMES ${libname}
    HINTS ${_TRT_HINT_ROOTS}
    PATH_SUFFIXES ${_TRT_LIB_SUFFIXES}
  )
endfunction()

_tensorrt_find_lib(TensorRT_nvinfer_LIBRARY        nvinfer)
_tensorrt_find_lib(TensorRT_nvinfer_plugin_LIBRARY nvinfer_plugin)
_tensorrt_find_lib(TensorRT_nvonnxparser_LIBRARY   nvonnxparser)

_tensorrt_find_lib(TensorRT_nvparsers_LIBRARY      nvparsers)

find_package_handle_standard_args(TensorRT
  REQUIRED_VARS
    TensorRT_INCLUDE_DIR
    TensorRT_nvinfer_LIBRARY
    TensorRT_nvinfer_plugin_LIBRARY
    TensorRT_nvonnxparser_LIBRARY
)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
  add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
  set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
  )
endif()

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer_plugin)
  add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
  set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
    IMPORTED_LOCATION "${TensorRT_nvinfer_plugin_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
  )
endif()

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvonnxparser)
  add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
  set_target_properties(TensorRT::nvonnxparser PROPERTIES
    IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
  )
endif()

if(TensorRT_FOUND AND TensorRT_nvparsers_LIBRARY AND NOT TARGET TensorRT::nvparsers)
  add_library(TensorRT::nvparsers UNKNOWN IMPORTED)
  set_target_properties(TensorRT::nvparsers PROPERTIES
    IMPORTED_LOCATION "${TensorRT_nvparsers_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
  )
endif()

set(TENSORRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
set(TENSORRT_LIBRARIES
  "${TensorRT_nvinfer_LIBRARY}"
  "${TensorRT_nvinfer_plugin_LIBRARY}"
  "${TensorRT_nvonnxparser_LIBRARY}"
)
if(TensorRT_nvparsers_LIBRARY)
  list(APPEND TENSORRT_LIBRARIES "${TensorRT_nvparsers_LIBRARY}")
endif()

mark_as_advanced(
  TensorRT_INCLUDE_DIR
  TensorRT_nvinfer_LIBRARY
  TensorRT_nvinfer_plugin_LIBRARY
  TensorRT_nvonnxparser_LIBRARY
  TensorRT_nvparsers_LIBRARY
)

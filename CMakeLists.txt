cmake_minimum_required(VERSION 3.16)

set(AI_DETECTION_NVCC "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "Path to nvcc")

if((NOT DEFINED CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER STREQUAL "") AND NOT DEFINED CUDACXX)

  if(EXISTS "${AI_DETECTION_NVCC}")
    set(CMAKE_CUDA_COMPILER "${AI_DETECTION_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
    message(STATUS "[ROOT] Using nvcc from AI_DETECTION_NVCC: ${CMAKE_CUDA_COMPILER}")
  else()
    find_program(_AI_DETECTION_NVCC
      NAMES nvcc
      PATHS
        /usr/local/cuda/bin
        /usr/local/cuda-12/bin
        /usr/local/cuda-11/bin
        /usr/bin
      NO_DEFAULT_PATH
    )

    if(NOT _AI_DETECTION_NVCC)
      find_program(_AI_DETECTION_NVCC NAMES nvcc)
    endif()

    if(_AI_DETECTION_NVCC)
      set(CMAKE_CUDA_COMPILER "${_AI_DETECTION_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
      message(STATUS "[ROOT] Found nvcc: ${CMAKE_CUDA_COMPILER}")
    else()
      message(STATUS "[ROOT] nvcc not found in common paths and not in PATH.")
    endif()
  endif()
endif()

project(ai-detection VERSION ${VERSION_NUMBER} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_INSTALL_RPATH "/usr/local/bin")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "[ROOT] Platform detected: Jetson/aarch64, setting CUDA arch to 87")
    set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures" FORCE)
  else()
    message(STATUS "[ROOT] Platform detected: Desktop/PC, setting CUDA arch to native")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
  endif()
endif()
message(STATUS "[ROOT] CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")

set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen_339/share/eigen3/cmake")
find_package(Eigen3 3.3.9 REQUIRED NO_MODULE)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)

add_subdirectory(ai_detection_cxx/botsort)
add_subdirectory(ai_detection_cxx/yolo)
add_subdirectory(ai_detection_cxx/bytetrack)
add_subdirectory(external/data-bus)
add_subdirectory(external/spdlog)
add_subdirectory(external/cxxopts)

add_executable(${PROJECT_NAME} ai_detection_cxx/main.cpp)

target_compile_definitions(${PROJECT_NAME} PRIVATE API_EXPORTS EIGEN_MPL2_ONLY)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/ai_detection_cxx/botsort/include
  ${EIGEN3_INCLUDE_DIRS}
  external/cxxopts/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  yolo
  botsort
  TensorRT::nvinfer
  TensorRT::nvinfer_plugin
  TensorRT::nvonnxparser
  ${OpenCV_LIBS}
  Eigen3::Eigen
  DataBus-cxx
  spdlog::spdlog
)

if(TARGET TensorRT::nvparsers)
  target_link_libraries(${PROJECT_NAME} PRIVATE TensorRT::nvparsers)
endif()

add_executable(webcam ai_detection_cxx/webcam_publisher.cpp)

target_include_directories(webcam PRIVATE
  external/data-bus/data_bus_cxx/types
  external/data-bus/data_bus_cxx/src
  external/cxxopts/include
)

target_link_libraries(webcam PRIVATE
  ddscxx
  opencv_core opencv_imgcodecs opencv_highgui opencv_videoio opencv_imgproc
  spdlog::spdlog
  DataBus-cxx
)

install(TARGETS ${PROJECT_NAME} webcam DESTINATION bin)
include(CMakeCPack.cmake)

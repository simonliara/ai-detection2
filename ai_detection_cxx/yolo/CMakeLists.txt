cmake_minimum_required(VERSION 3.16)

project(yolo VERSION 1.0 LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME} STATIC
  src/YOLOv11.cpp
  src/preprocess.cu
  src/postprocess.cu
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_STANDARD 14
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS}
  CUDA::cudart
  TensorRT::nvinfer
  TensorRT::nvinfer_plugin
  TensorRT::nvonnxparser
  spdlog::spdlog
  botsort
  bytetrack
)

if(TARGET TensorRT::nvparsers)
  target_link_libraries(${PROJECT_NAME} PUBLIC TensorRT::nvparsers)
endif()

add_executable(onnx2engine src/onnx2engine.cpp)

target_link_libraries(onnx2engine PRIVATE
  CUDA::cudart
  TensorRT::nvinfer
  TensorRT::nvinfer_plugin
  TensorRT::nvonnxparser
  spdlog::spdlog
)

if(TARGET TensorRT::nvparsers)
  target_link_libraries(onnx2engine PRIVATE TensorRT::nvparsers)
endif()

set_target_properties(onnx2engine PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

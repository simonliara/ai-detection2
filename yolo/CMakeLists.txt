cmake_minimum_required(VERSION 3.16)

project(yolo VERSION 1.0 LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(STATUS "[YOLO] Platform detected: NVIDIA Jetson")
        set(CMAKE_CUDA_ARCHITECTURES 87) 
    else()
        message(STATUS "[YOLO] Platform detected: Desktop/PC")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()
message(STATUS "[YOLO] Setting CUDA Architecture to: ${CMAKE_CUDA_ARCHITECTURES}")

add_library(${PROJECT_NAME} SHARED
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

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  ${TENSORRT_INCLUDE_DIR} 
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  botsort
  ${OpenCV_LIBS}
  CUDA::cudart
  ${TENSORRT_LIBS}       
)

add_executable(onnx2engine src/onnx2engine.cpp)

target_include_directories(onnx2engine PRIVATE
    ${TENSORRT_INCLUDE_DIR} 
)

target_link_libraries(onnx2engine PRIVATE
    CUDA::cudart
    ${TENSORRT_LIBS}       
)

set_target_properties(onnx2engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
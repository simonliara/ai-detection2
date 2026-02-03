cmake_minimum_required(VERSION 3.16)

if(NOT CMAKE_CUDA_COMPILER)
    if(EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    endif()
endif()

set(VERSION_NUMBER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/version.txt")
if(EXISTS "${VERSION_NUMBER_FILE}")
    file(STRINGS "${VERSION_NUMBER_FILE}" VERSION_NUMBER)
else()
    set(VERSION_NUMBER "0.1.0")
endif()

project(ai-detection VERSION ${VERSION_NUMBER} LANGUAGES CXX CUDA)

set(CMAKE_INSTALL_RPATH "/usr/local/bin")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release MinSizeRel RelWithDebInfo)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen_339/share/eigen3/cmake")
find_package(Eigen3 3.3.9 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(TENSORRT_LIB_NAMES nvinfer nvinfer_plugin nvonnxparser)
set(TENSORRT_LIBS "")

set(TRT_SEARCH_PATHS
    /usr/local/TensorRT-8.6.1    
    /usr/local/tensorrt        
    /usr                     
    /usr/local                  
)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR} $ENV{TENSORRT_DIR}
    PATHS ${TRT_SEARCH_PATHS}
    PATH_SUFFIXES include include/aarch64-linux-gnu include/x86_64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT headers not found. Use -DTENSORRT_DIR=/path/to/trt")
endif()

foreach(LIB ${TENSORRT_LIB_NAMES})
    find_library(TRT_LIB_${LIB} ${LIB}
        HINTS ${TENSORRT_DIR} $ENV{TENSORRT_DIR}
        PATHS ${TRT_SEARCH_PATHS}
        PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu
    )
    if(NOT TRT_LIB_${LIB})
        message(FATAL_ERROR "TensorRT Library '${LIB}' not found.")
    endif()
    list(APPEND TENSORRT_LIBS ${TRT_LIB_${LIB}})
endforeach()

find_library(TRT_LIB_nvparsers nvparsers
    HINTS ${TENSORRT_DIR} $ENV{TENSORRT_DIR}
    PATHS ${TRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu
)
if(TRT_LIB_nvparsers)
    list(APPEND TENSORRT_LIBS ${TRT_LIB_nvparsers})
endif()

add_subdirectory(ai_detection_cxx/botsort)
add_subdirectory(ai_detection_cxx/yolo)
add_subdirectory(ai_detection_cxx/bytetrack)
add_subdirectory(external/data-bus)
add_subdirectory(external/spdlog)

add_executable(${PROJECT_NAME} ai_detection_cxx/main.cpp)

target_compile_definitions(${PROJECT_NAME} PRIVATE API_EXPORTS EIGEN_MPL2_ONLY)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${TENSORRT_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/ai_detection_cxx/botsort/include
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  yolo
  botsort
  ${TENSORRT_LIBS}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  DataBus-cxx
  spdlog::spdlog
)

add_executable(webcam ai_detection_cxx/webcam_publisher.cpp)

target_include_directories(webcam PRIVATE
    external/data-bus/data_bus_cxx/types
    external/data-bus/data_bus_cxx/src
)

target_link_libraries(webcam PRIVATE
    ddscxx
    opencv_core opencv_imgcodecs opencv_highgui opencv_videoio opencv_imgproc
    spdlog::spdlog
    DataBus-cxx
)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
include(CMakeCPack.cmake)
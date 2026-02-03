cmake_minimum_required(VERSION 3.14)
project(bytetrack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DCOMPILEDWITHC17)

include(FetchContent)
set(EIGEN_VERSION 3.3)
find_package(Eigen3 ${EIGEN_VERSION} QUIET)
if(NOT EIGEN3_FOUND)
  message(STATUS "Eigen3 not found locally, fetching from git...")
  FetchContent_Declare(eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
    GIT_TAG         3.3.9
    GIT_SHALLOW     ON)
  FetchContent_MakeAvailable(eigen)
endif()

add_library(${PROJECT_NAME} STATIC
  src/BYTETracker.cpp
  src/KalmanFilter.cpp
  src/lapjv.cpp
  src/Object.cpp
  src/Rect.cpp
  src/STrack.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(TARGET Eigen3::Eigen)
  target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
else()
  target_link_libraries(${PROJECT_NAME} PUBLIC eigen)
endif()

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -O2)
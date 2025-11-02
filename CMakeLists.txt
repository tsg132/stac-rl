cmake_minimum_required(VERSION 3.18)
project(STAC-RL VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler options
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/cuda)

# Source files
file(GLOB MODEL_SOURCES "src/model/*.cpp")
file(GLOB TRAINING_SOURCES "src/training/*.cpp")
file(GLOB ENV_SOURCES "src/env/*.cpp")
file(GLOB MAIN_SOURCES "src/main_infer.cpp")

set(SOURCES ${MODEL_SOURCES} ${TRAINING_SOURCES} ${ENV_SOURCES})

# Add CUDA subdirectory first
add_subdirectory(cuda)

# Create main library
add_library(stac_core STATIC ${SOURCES})

target_include_directories(stac_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/cuda
)

target_link_libraries(stac_core PUBLIC
    stac_cuda
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(stac_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(stac_core PUBLIC USE_OPENMP)
endif()

# Check if CUDA was actually built (not just an interface library)
get_target_property(CUDA_TYPE stac_cuda TYPE)
if(NOT CUDA_TYPE STREQUAL "INTERFACE_LIBRARY")
    target_compile_definitions(stac_core PUBLIC USE_CUDA)
    message(STATUS "Building with CUDA support")
else()
    message(STATUS "Building CPU-only version (CUDA disabled)")
endif()

# Create inference executable
add_executable(stac_infer ${MAIN_SOURCES})
target_link_libraries(stac_infer PRIVATE stac_core)

# Create training executable
add_executable(stac_train src/main_train.cpp)
target_link_libraries(stac_train PRIVATE stac_core)

message(STATUS "STAC-RL project configured successfully")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX Flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")

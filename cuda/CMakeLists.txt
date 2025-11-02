cmake_minimum_required(VERSION 3.18)

# Try to enable CUDA language
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER AND USE_CUDA)
    enable_language(CUDA)
    
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Set CUDA architectures (can be overridden)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)  # Default to T4
    endif()
    
    # Find CUDA toolkit
    find_package(CUDAToolkit REQUIRED)
    
    # CUDA source files
    set(CUDA_SOURCES
        flash_attention_kernel.cu
        cuda_ops.cu
        cuda_wrappers.cu
    )
    
    # Create CUDA library
    add_library(stac_cuda STATIC ${CUDA_SOURCES})
    
    target_include_directories(stac_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )
    
    target_link_libraries(stac_cuda PUBLIC
        CUDA::cudart
        CUDA::cublas
    )
    
    # Set CUDA compilation flags
    target_compile_options(stac_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            --expt-relaxed-constexpr
            -O3
        >
    )
    
    message(STATUS "CUDA enabled")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA not found or disabled - building CPU-only version")
    
    # Create empty library for CPU-only builds
    add_library(stac_cuda INTERFACE)
endif()

cmake_minimum_required(VERSION 3.18)

find_package(CUDA QUIET)

if(CUDA_FOUND)
    enable_language(CUDA)
    
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Turing, Ampere, Ada
    
    # CUDA source files
    set(CUDA_SOURCES
        flash_attention_kernel.cu
        cuda_ops.cu
    )
    
    # Create CUDA library
    add_library(stac_cuda STATIC ${CUDA_SOURCES})
    
    target_include_directories(stac_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
    )
    
    target_link_libraries(stac_cuda PUBLIC
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
    )
    
    # Set CUDA compilation flags
    target_compile_options(stac_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            --expt-relaxed-constexpr
            -O3
        >
    )
    
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    message(STATUS "CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
else()
    message(STATUS "CUDA not found - building CPU-only version")
    
    # Create empty library for CPU-only builds
    add_library(stac_cuda INTERFACE)
endif()

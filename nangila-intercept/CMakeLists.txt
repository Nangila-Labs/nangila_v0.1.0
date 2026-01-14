cmake_minimum_required(VERSION 3.18)
project(nangila-intercept LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Nangila Rust static library path
set(NANGILA_LIB_DIR "${CMAKE_SOURCE_DIR}/../target/release" CACHE PATH "Nangila lib directory")
set(NANGILA_LIB_NAME "nangila" CACHE STRING "Nangila library name")

# Find the static library
find_library(NANGILA_LIBRARY
    NAMES lib${NANGILA_LIB_NAME}.a ${NANGILA_LIB_NAME}
    PATHS ${NANGILA_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT NANGILA_LIBRARY)
    message(WARNING "Nangila static library not found in ${NANGILA_LIB_DIR}")
    message(WARNING "Run 'cargo build --release -p nangila-hook' first")
    # Create a placeholder so cmake can complete
    set(NANGILA_LIBRARY "${NANGILA_LIB_DIR}/libnangila.a")
endif()

# Create the intercept shared library
add_library(nangila_intercept SHARED
    intercept.cpp
)

target_include_directories(nangila_intercept PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nangila_intercept PRIVATE
    ${NANGILA_LIBRARY}
    CUDA::cudart
    dl
    pthread
)

# System libraries needed by Rust
if(APPLE)
    target_link_libraries(nangila_intercept PRIVATE
        "-framework Security"
        "-framework CoreFoundation"
    )
else()
    target_link_libraries(nangila_intercept PRIVATE
        m
    )
endif()

# Install
install(TARGETS nangila_intercept
    LIBRARY DESTINATION lib
)

install(FILES include/nangila.h
    DESTINATION include
)

# Print configuration
message(STATUS "Nangila library: ${NANGILA_LIBRARY}")
message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")

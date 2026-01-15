cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(nangila_ddp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Nangila FFI library location (built via cargo)
set(NANGILA_LIB_DIR "${CMAKE_SOURCE_DIR}/../target/release")

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

# Create the Python extension module
pybind11_add_module(nangila_ddp_cpp
    src/bindings.cpp
)

# Link against libtorch and nangila
target_link_libraries(nangila_ddp_cpp PRIVATE
    ${TORCH_LIBRARIES}
    ${NANGILA_LIB_DIR}/libnangila_hook.a
    CUDA::cudart
    pthread
    dl
)

# Set rpath for finding libtorch at runtime
set_target_properties(nangila_ddp_cpp PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

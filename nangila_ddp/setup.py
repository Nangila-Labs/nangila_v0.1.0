"""
Nangila C++ DDP Hook - Build Script

This script builds the native C++ DDP hook using torch.utils.cpp_extension.
It requires:
- PyTorch with C++ extensions support
- CUDA toolkit
- Rust toolchain (for building libnangila_hook)
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# Get the directory containing this script
ROOT_DIR = Path(__file__).parent.resolve()
NANGILA_DIR = ROOT_DIR.parent
RUST_LIB_DIR = NANGILA_DIR / "target" / "release"

def build_rust_lib():
    """Build the Rust nangila-hook static library if not already built."""
    lib_path = RUST_LIB_DIR / "libnangila.a"
    
    # Skip if already built
    if lib_path.exists():
        print(f"Rust library already exists at {lib_path}")
        return
    
    print("Building Rust nangila-hook library...")
    env = os.environ.copy()
    
    # Try common cargo locations
    cargo_paths = [
        os.path.expanduser("~/.cargo/bin/cargo"),
        "/root/.cargo/bin/cargo",
        "cargo"
    ]
    
    cargo_cmd = None
    for path in cargo_paths:
        if os.path.exists(path) or path == "cargo":
            cargo_cmd = path
            break
    
    if cargo_cmd is None:
        print("WARNING: cargo not found, skipping Rust build")
        print("Please build manually: cargo build --release -p nangila-hook")
        return
    
    result = subprocess.run(
        [cargo_cmd, "build", "--release", "-p", "nangila-hook"],
        cwd=NANGILA_DIR,
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Rust build failed:\n{result.stderr}")
        print("Please build manually: cargo build --release -p nangila-hook")
        return
    
    print("Rust library built successfully")

# Build Rust first (will skip if already built)
build_rust_lib()


# Paths
INCLUDE_DIR = ROOT_DIR / "include"
SRC_DIR = ROOT_DIR / "src"

# Extension module
ext_modules = [
    CUDAExtension(
        name="nangila_ddp_cpp",
        sources=[
            str(SRC_DIR / "bindings.cpp"),
        ],
        include_dirs=[
            str(INCLUDE_DIR),
        ],
        library_dirs=[
            str(RUST_LIB_DIR),
        ],
        libraries=[
            "nangila",
        ],
        extra_compile_args={
            "cxx": ["-std=c++17", "-O3"],
            "nvcc": ["-std=c++17", "-O3"],
        },
        extra_link_args=[
            f"-L{RUST_LIB_DIR}",
            "-lnangila",
            "-lpthread",
            "-ldl",
        ],
    )
]

setup(
    name="nangila_ddp_cpp",
    version="0.1.0",
    author="Nangila Team",
    description="Native C++ DDP hook for Nangila gradient compression",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)

#!/bin/bash
#
# Build the Nangila NCCL intercept library
#
# Usage:
#   ./build.sh [clean|release|debug]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
BUILD_TYPE="${1:-release}"

if [ "$BUILD_TYPE" = "clean" ]; then
    echo "Cleaning build artifacts..."
    rm -rf "$SCRIPT_DIR/build"
    rm -f "$SCRIPT_DIR/libnangila_intercept.so"
    exit 0
fi

CMAKE_BUILD_TYPE="Release"
if [ "$BUILD_TYPE" = "debug" ]; then
    CMAKE_BUILD_TYPE="Debug"
fi

echo "========================================"
echo "Building Nangila NCCL Intercept Library"
echo "========================================"
echo ""

# Step 1: Build Rust library
echo "[1/3] Building Rust library..."
cd "$ROOT_DIR"

if [ "$BUILD_TYPE" = "debug" ]; then
    cargo build -p nangila-hook
    LIB_DIR="$ROOT_DIR/target/debug"
else
    cargo build --release -p nangila-hook
    LIB_DIR="$ROOT_DIR/target/release"
fi

if [ ! -f "$LIB_DIR/libnangila.a" ]; then
    echo "ERROR: Static library not found at $LIB_DIR/libnangila.a"
    echo "Make sure cargo build completed successfully"
    exit 1
fi

echo "  Rust library: $LIB_DIR/libnangila.a"

# Step 2: Configure CMake
echo ""
echo "[2/3] Configuring CMake..."
cd "$SCRIPT_DIR"
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DNANGILA_LIB_DIR="$LIB_DIR"

# Step 3: Build
echo ""
echo "[3/3] Building intercept library..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Copy to root of intercept directory for easy access
cp libnangila_intercept.so "$SCRIPT_DIR/"

echo ""
echo "========================================"
echo "Build complete!"
echo "========================================"
echo ""
echo "Output: $SCRIPT_DIR/libnangila_intercept.so"
echo ""
echo "Usage:"
echo "  LD_PRELOAD=$SCRIPT_DIR/libnangila_intercept.so \\"
echo "  NANGILA_MASK=topology.nzmask \\"
echo "  torchrun --nproc_per_node=8 train.py"
echo ""

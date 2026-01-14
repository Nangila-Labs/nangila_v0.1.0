#!/bin/bash
# Full 8-GPU Validation Suite
# Run all tests and collect results

set -e

echo "=========================================="
echo "Nangila Full 8-GPU Validation"
echo "=========================================="
echo "Start time: $(date)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

cd "$SCRIPT_DIR/.."

# Test 1: Rust unit tests
echo ""
echo "=========================================="
echo "Test 1: Rust Unit Tests"
echo "=========================================="
cargo test --workspace --exclude nangila-cuda 2>&1 | tee "$RESULTS_DIR/rust_tests.log"
echo "✓ Rust tests complete"

# Test 2: Bandwidth benchmark
echo ""
echo "=========================================="
echo "Test 2: Bandwidth Benchmark (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 scripts/bandwidth_benchmark.py 2>&1 | tee "$RESULTS_DIR/bandwidth.log"
echo "✓ Bandwidth benchmark complete"

# Test 3: Baseline training (no Nangila)
echo ""
echo "=========================================="
echo "Test 3: Baseline Training (100 steps)"
echo "=========================================="
torchrun --nproc_per_node=8 scripts/train_llm.py \
    --model ./mistral-7b \
    --steps 100 \
    --batch-size 2 \
    2>&1 | tee "$RESULTS_DIR/baseline_100.log"
echo "✓ Baseline training complete"

# Test 4: Nangila τ=0.97 (conservative)
echo ""
echo "=========================================="
echo "Test 4: Nangila τ=0.97 (100 steps)"
echo "=========================================="
torchrun --nproc_per_node=8 scripts/train_llm.py \
    --model ./mistral-7b \
    --nangila \
    --threshold 0.97 \
    --steps 100 \
    --batch-size 2 \
    2>&1 | tee "$RESULTS_DIR/nangila_097.log"
echo "✓ Nangila τ=0.97 complete"

# Test 5: Nangila τ=0.95 (default)
echo ""
echo "=========================================="
echo "Test 5: Nangila τ=0.95 (100 steps)"
echo "=========================================="
torchrun --nproc_per_node=8 scripts/train_llm.py \
    --model ./mistral-7b \
    --nangila \
    --threshold 0.95 \
    --steps 100 \
    --batch-size 2 \
    2>&1 | tee "$RESULTS_DIR/nangila_095.log"
echo "✓ Nangila τ=0.95 complete"

# Test 6: Nangila τ=0.90 (aggressive)
echo ""
echo "=========================================="
echo "Test 6: Nangila τ=0.90 (100 steps)"
echo "=========================================="
torchrun --nproc_per_node=8 scripts/train_llm.py \
    --model ./mistral-7b \
    --nangila \
    --threshold 0.90 \
    --steps 100 \
    --batch-size 2 \
    2>&1 | tee "$RESULTS_DIR/nangila_090.log"
echo "✓ Nangila τ=0.90 complete"

# Test 7: Longer convergence test
echo ""
echo "=========================================="
echo "Test 7: Convergence Test (1000 steps)"
echo "=========================================="
echo "Running baseline..."
torchrun --nproc_per_node=8 scripts/train_llm.py \
    --model ./mistral-7b \
    --steps 1000 \
    --batch-size 2 \
    2>&1 | tee "$RESULTS_DIR/convergence_baseline.log"

echo "Running Nangila..."
torchrun --nproc_per_node=8 scripts/train_llm.py \
    --model ./mistral-7b \
    --nangila \
    --threshold 0.95 \
    --steps 1000 \
    --batch-size 2 \
    2>&1 | tee "$RESULTS_DIR/convergence_nangila.log"
echo "✓ Convergence tests complete"

# Summary
echo ""
echo "=========================================="
echo "VALIDATION COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Result files:"
ls -la "$RESULTS_DIR"
echo ""

# Collect JSON results
echo "JSON Results:"
cat results_*.json 2>/dev/null || echo "No JSON results found"

echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR"
echo "  2. Compare loss curves between baseline and Nangila"
echo "  3. Update README with benchmark numbers"

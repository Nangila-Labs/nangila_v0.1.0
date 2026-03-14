# Nangila-SPICE: A Unified Framework for Chip Verification

**Addendum to Nangila Whitepaper v2**

---

## 1. Use Case: Nangila-SPICE for Chip Verification

As Moore's Law slows, the industry is shifting to specialized architectures like Compute-in-Memory (CiM). Verifying these billion-transistor mixed-signal designs is currently intractable with traditional SPICE.

### 1.1 The "Predict-Verify" Solver
Traditional SPICE relies on global synchronization at every time step. Nangila-SPICE replaces this with **Predictive Partitioning**:
*   **Method**: The circuit is split into thousands of partitions. Each partition *predicts* the boundary voltages of its neighbors and solves its local matrix speculatively.
*   **Result**: Inter-partition communication latency is hidden. Parallel scaling becomes linear rather than saturating at ~64 cores.

### 1.2 PVT-Space Acceleration
Verification requires simulating thousands of Process, Voltage, and Temperature (PVT) corners.
*   **Concept**: We treat the "Nominal" corner as a predictor for the "Slow" corner.
*   **Efficiency**: The solver only computes the residual difference between corners, converging 2-5x faster per simulation.

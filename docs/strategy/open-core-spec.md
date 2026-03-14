# Open-Core Specification: Unified State-Space Framework

**Reference**: [../archive/whitepaper-v2.md](../archive/whitepaper-v2.md)

This document defines the functional requirements and execution roadmap for the **Open-Core "Mini-Nangila"** release. It delineates the boundary between the community edition (verifiable science) and the enterprise edition (industrial scale).

---

## 1. Core Philosophy: The "Science vs. Scale" Split

The Open-Core release must serve as a **standalone, verifiable reference implementation** that proves the mathematical validity of the **Predictive-Residual** approach across all three domains (AI, HPC, Digital Twins).

| Feature Category | Open-Core (Mini-Nangila) | Proprietary (Enterprise) |
| :--- | :--- | :--- |
| **Goal** | Verify Entropy Reduction & Error Bounds | Maximize Scale & Real-Time Control |
| **Scale** | Single-Node / Small Cluster (2-8 GPUs) | Extreme Scale (100+ Nodes / Exascale) |
| **Predictors** | Generic (Momentum, Linear Extrapolator) | Domain-Specific (Navier-Stokes, FEA) |
| **Transport** | TCP / Gloo | MPI (InfiniBand), 5G/Satellite Optimization |
| **Safety** | Basic Error Checks | ISO 26262 / DO-178C Compliance |

---

## 2. Functional Requirements (What It Should Do)

### 2.1 Shared Core (The "Math" Layer)
*ALL domains use these foundational components.*

-   **Deterministic Arithmetic (Q8.23)**:
    -   **Bit-Exactness**: Must produce identical results on x86_64/ARM64 and CUDA/ROCm.
    -   **FixedPointBuffer**: Range $[-128, 127.999999]$, Precision $2^{-23}$.
    -   **Validation**: Cross-platform unit tests guaranteeing determinism.
-   **Predictive-Residual Engine**:
    -   Abstract `Predictor` trait (`predict(state_history) -> prediction`).
    -   Abstract `Quantizer` trait (`quantize(residual) -> (compressed, error)`).

### 2.2 Domain A: AI Training (Stochastic Mode)
*Focus: Convergence & Compression*

| Component | Open-Core Spec | Enterprise Add-on |
| :--- | :--- | :--- |
| **Predictor** | **Momentum (EMA)**: $\hat{g}_t = \beta \hat{g}_{t-1} + (1-\beta)g_{t-1}$ | **Learned**: Neural predictors, transformer-based |
| **Quantizer** | **Stochastic INT4**: Standard randomized rounding | **Adaptive**: Bit-rate control per layer |
| **Topology** | **Static Masking**: Drop fixed % of layers | **Dynamic Sculptor**: Real-time layer promotion |
| **Scale** | **DDP**: PyTorch Distributed Data Parallel | **FSDP / MoE**: Sharded parameters, Expert routing |

### 2.3 Domain B: HPC Simulation (Deterministic Mode)
*Focus: Stability & Checkpointing*

| Component | Open-Core Spec | Enterprise Add-on |
| :--- | :--- | :--- |
| **Predictor** | **Linear Extrapolator**: $\hat{S}_{t+1} = S_t + \Delta t \cdot \dot{S}_t$ | **Physics-Informed**: Plug-in hooks for OpenFOAM/Solver state |
| **Quantizer** | **Error-Bounded (SZ-Basic)**: Guarantee $\|x - \hat{x}\| < \epsilon$ | **Advanced**: curve-fitting, variable-rate ZFP |
| **Use Case** | **Checkpoint Compression**: Single-node HDF5 writer | **Predictive Halo**: MPI-native ghost cell exchange |
| **Safety** | **Epsilon Check**: Fallback to lossless if $err > \epsilon$ | **Rollback**: Distributed state recovery for speculative exec |

### 2.4 Domain C: Digital Twins (Edge-Cloud)
*Focus: Telemetry & Synchronization*

| Component | Open-Core Spec | Enterprise Add-on |
| :--- | :--- | :--- |
| **Runtime** | **Rust/C Lib**: Compile for generic Linux | **Embedded**: Bare-metal ARM/RISC-V, RTOS support |
| **Sync** | **Unidirectional**: Edge $\to$ Cloud telemetry | **Bidirectional**: Real-time Control Loop (Cloud $\to$ Edge) |
| **Transport** | **gRPC / WebSocket**: Standard protocols | **Custom UDP**: Reliability over lossy 5G/Sat links |
| **Security** | **None**: Standard TLS assumed | **DePIN/Audit**: Cryptographic Proof of Simulation |

---

## 3. Reference Implementation Goals

The "Mini-Nangila" repo must provide working examples for:

1.  **AI**: Train `NanoGPT` (125M) on 2 GPUs with 20x compression vs. DDP.
2.  **HPC**: Compress a **LAMMPS** molecular dynamics trajectory with 50x ratio and verified $\epsilon$ error bound.
3.  **Twin**: Sync a simulated "spinning rotor" sensor stream from a Raspberry Pi (or emulated edge) to a PC with 100x bandwidth reduction.

---

## 4. Execution Roadmap (Open-Core)

### Phase 1: Core Consolidation (Weeks 1-4)
- [ ] **Dual-Mode Quantizer**: Refactor `Quantizer` to support `Stochastic` (AI) and `ErrorBounded` (HPC) modes.
- [ ] **Unified Predictor Trait**: Define common interface for Momentum (AI) and Linear (HPC) predictors.
- [ ] **Q8.23 Hardening**: Finalize fixed-point math library and publish as standalone crate `nangila-math`.

### Phase 2: HPC/Twin Modules (Weeks 5-8)
- [ ] **`nangila-checkpoint`**: Create HDF5 filter plugin for easy integration with scientific codes.
- [ ] **`nangila-edge`**: minimal `no_std` compatible subset of core for edge devices.
- [ ] **Benchmarks**:
    -   Proprietary: 175B LLM, Full Car CFD.
    -   **Open**: 1B LLM, Lennard-Jones MD, Simple Rotor Twin.

### Phase 3: Documentation & Release (Weeks 9-12)
- [ ] **Whitepaper Alignment**: Ensure docs match `archive/whitepaper-v2.md` terminology.
- [ ] **"Science" Tutorials**:
    -   *"How to verify the Entropy Reduction Lemma"*
    -   *"Reproducing the 16-day training result (scaled down)"*
    -   *"Why Determinism Matters for Safety"*
- [ ] **Public Release**: v0.1.0 on GitHub/PyPI/Crates.io.

---

## 5. Licensing Strategy

-   **Open Core (`nangila-core`)**: **Apache 2.0 / MIT**. Maximizes adoption, allows use in academic/commercial projects.
-   **Enterprise Plugins (`nangila-mpi`, `nangila-control`, `nangila-compliance`)**: **Proprietary Commercial License**.
-   **Contributor License Agreement (CLA)**: Required for contributions to core to ensure dual-licensing capability.

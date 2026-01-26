/*
 * Nangila CUDA Kernels: Fused Predict-Subtract-Quantize
 *
 * This kernel fuses three operations into one memory-bandwidth-optimal pass:
 * 1. Predict: ĝ = g_t + μ * (g_t - g_{t-1})
 * 2. Subtract: r = g - ĝ
 * 3. Quantize: clamp(stochastic_round(r / γ), -8, 7)
 *
 * Memory access pattern:
 * - Read: gradient (current), g_t, g_{t-1}
 * - Write: packed INT4 output
 *
 * All intermediate values stay in registers = minimal memory bandwidth.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Configuration
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Data type enum for kernel selection
enum NangilaDtype { DTYPE_FLOAT32 = 0, DTYPE_FLOAT16 = 1, DTYPE_BFLOAT16 = 2 };

// INT4 packing: two values per byte
// Low nibble = first value, high nibble = second value
__device__ __forceinline__ uint8_t pack_int4(int8_t a, int8_t b) {
  return ((uint8_t)(a & 0x0F)) | (((uint8_t)(b & 0x0F)) << 4);
}

// Conversion helpers for half precision
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

// -----------------------------------------------------------------------------
// Stochastic Rounding Logic (Matches Rust Implementation)
// -----------------------------------------------------------------------------

// MurmurHash3-like mixer for deterministic pseudo-randomness
__device__ __forceinline__ uint64_t hash_prng(uint64_t seed, uint64_t step,
                                              uint32_t layer_id,
                                              uint32_t index) {
  // Constants from Rust implementation (SplitMix64 / MurmurHash3 constants)
  const uint64_t K1 = 0x9E3779B97F4A7C15ULL;
  const uint64_t K2 = 0xBF58476D1CE4E5B9ULL;
  const uint64_t K3 = 0x94D049BB133111EBULL;

  // Build hash base from layer_id + step (same across all ranks)
  uint64_t hash_base = (uint64_t)layer_id * K1 + step * K2;

  // Mix with global index and seed (using fixed seed 42 as per default in Rust)
  uint64_t hash_input = seed * K1 + hash_base + index;

  uint64_t h = hash_input;
  h ^= (h >> 30);
  h *= K2;
  h ^= (h >> 27);
  h *= K3;
  h ^= (h >> 31);

  return h;
}

// Stochastic rounding function
__device__ __forceinline__ int8_t stochastic_round(float val, uint64_t step,
                                                   uint32_t layer_id,
                                                   uint32_t global_idx) {
  float floor_val = floorf(val);
  float frac = fabsf(val - floor_val);

  // Generate random float in [0, 1) using the hash
  uint64_t rng =
      hash_prng(42, step, layer_id, global_idx); // Seed 42 matches Rust default
  float random_01 = (float)rng / (float)0xFFFFFFFFFFFFFFFFULL;

  int8_t res;
  if (frac >= random_01) {
    // Round away from zero (magnitude matches ceil of abs)
    // If val >= 0: floor + 1
    // If val < 0: floor (since floor is already more negative, we want closer
    // to zero??) Wait, let's match Rust implementation exactly:
    /*
        let scaled = x / gamma;
        let floor = scaled.floor();
        let frac = (scaled - floor).abs();

        if frac >= random_01 {
            if scaled >= 0.0 { (floor + 1.0) as i8 } else { floor as i8 }
        } else {
            if scaled >= 0.0 { floor as i8 } else { (floor + 1.0) as i8 }
        }
    */
    if (val >= 0.0f) {
      res = (int8_t)(floor_val + 1.0f);
    } else {
      res = (int8_t)floor_val;
    }
  } else {
    if (val >= 0.0f) {
      res = (int8_t)floor_val;
    } else {
      res = (int8_t)(floor_val + 1.0f);
    }
  }
  return res;
}

/*
 * Fused Predict-Subtract-Quantize Kernel
 *
 * Inputs:
 *   gradient:    Current gradient to compress [N]
 *   g_current:   Gradient from step t (for prediction) [N]
 *   g_previous:  Gradient from step t-1 (for prediction) [N]
 *   momentum:    Momentum coefficient (typically 0.9)
 *   gamma:       Quantization scale factor
 *   N:           Number of elements
 *   step:        Current training step (for RNG)
 *   layer_id:    Layer ID (for RNG)
 *
 * Outputs:
 *   output:      Packed INT4 quantized residuals [N/2]
 */
extern "C" __global__ void predict_subtract_quantize_kernel(
    const float *__restrict__ gradient, const float *__restrict__ g_current,
    const float *__restrict__ g_previous, float momentum, float gamma,
    uint8_t *__restrict__ output, int N, uint64_t step, uint32_t layer_id) {
  // Each thread handles 2 elements (packed into 1 byte)
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  if (idx >= N)
    return;

  // Process first element
  float g0 = gradient[idx];
  float curr0 = g_current[idx];
  float prev0 = g_previous[idx];

  // Predict: ĝ = g_t + μ * (g_t - g_{t-1})
  float pred0 = curr0 + momentum * (curr0 - prev0);

  // Subtract: r = g - ĝ
  float residual0 = g0 - pred0;

  // Quantize: clamp(stochastic_round(r / γ), -8, 7)
  float scaled0 = residual0 / gamma;
  int8_t quant0 = stochastic_round(scaled0, step, layer_id, idx);
  quant0 = max(-8, min(7, (int)quant0));

  // Process second element (if within bounds)
  int8_t quant1 = 0;
  if (idx + 1 < N) {
    float g1 = gradient[idx + 1];
    float curr1 = g_current[idx + 1];
    float prev1 = g_previous[idx + 1];

    float pred1 = curr1 + momentum * (curr1 - prev1);
    float residual1 = g1 - pred1;
    float scaled1 = residual1 / gamma;
    int8_t raw_quant1 = stochastic_round(scaled1, step, layer_id, idx + 1);
    quant1 = max(-8, min(7, (int)raw_quant1));
  }

  // Pack two INT4 values into one byte
  output[idx / 2] = pack_int4(quant0, quant1);
}

/*
 * Vectorized version using float4 for better memory coalescing
 * Processes 8 elements per thread (4 output bytes)
 */
extern "C" __global__ void predict_subtract_quantize_vec4_kernel(
    const float4 *__restrict__ gradient, const float4 *__restrict__ g_current,
    const float4 *__restrict__ g_previous, float momentum, float gamma,
    uint32_t *__restrict__ output, // 4 bytes = 8 INT4 values
    int N4,                        // N / 4
    uint64_t step, uint32_t layer_id) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx * 2 >= N4)
    return;

  // Load 8 floats (2 float4s)
  float4 g_a = gradient[idx * 2];
  float4 g_b = gradient[idx * 2 + 1];
  float4 curr_a = g_current[idx * 2];
  float4 curr_b = g_current[idx * 2 + 1];
  float4 prev_a = g_previous[idx * 2];
  float4 prev_b = g_previous[idx * 2 + 1];

  float inv_gamma = 1.0f / gamma;

  // Process all 8 elements
  int8_t q[8];

  // Base index for RNG
  int base_idx =
      idx * 8; // idx is thread index in N4 space (vec8 space), so * 8

#define PROCESS(i, ga, ca, pa)                                                 \
  do {                                                                         \
    float pred = ca + momentum * (ca - pa);                                    \
    float residual = ga - pred;                                                \
    float scaled = residual * inv_gamma;                                       \
    int8_t raw = stochastic_round(scaled, step, layer_id, base_idx + i);       \
    q[i] = (int8_t)max(-8, min(7, (int)raw));                                  \
  } while (0)

  PROCESS(0, g_a.x, curr_a.x, prev_a.x);
  PROCESS(1, g_a.y, curr_a.y, prev_a.y);
  PROCESS(2, g_a.z, curr_a.z, prev_a.z);
  PROCESS(3, g_a.w, curr_a.w, prev_a.w);
  PROCESS(4, g_b.x, curr_b.x, prev_b.x);
  PROCESS(5, g_b.y, curr_b.y, prev_b.y);
  PROCESS(6, g_b.z, curr_b.z, prev_b.z);
  PROCESS(7, g_b.w, curr_b.w, prev_b.w);

#undef PROCESS

  // Pack 8 INT4 values into 4 bytes (uint32_t)
  uint32_t packed = 0;
  packed |= ((uint32_t)(q[0] & 0x0F)) << 0;
  packed |= ((uint32_t)(q[1] & 0x0F)) << 4;
  packed |= ((uint32_t)(q[2] & 0x0F)) << 8;
  packed |= ((uint32_t)(q[3] & 0x0F)) << 12;
  packed |= ((uint32_t)(q[4] & 0x0F)) << 16;
  packed |= ((uint32_t)(q[5] & 0x0F)) << 20;
  packed |= ((uint32_t)(q[6] & 0x0F)) << 24;
  packed |= ((uint32_t)(q[7] & 0x0F)) << 28;

  output[idx] = packed;
}

/*
 * Synchronization modes for error checking
 * 0 = Async: No sync, maximum performance (production)
 * 1 = Always: Always sync, catch all errors (debug)
 * 2 = Periodic: Sync every 100 calls (balanced)
 */
enum SyncMode { SYNC_ASYNC = 0, SYNC_ALWAYS = 1, SYNC_PERIODIC = 2 };

/*
 * Host wrapper function with comprehensive error checking
 * Returns cudaError_t for error handling by caller
 */
extern "C" cudaError_t launch_predict_subtract_quantize(
    const float *gradient, const float *g_current, const float *g_previous,
    float momentum, float gamma, uint8_t *output, int N, cudaStream_t stream,
    int sync_mode, uint64_t step, uint32_t layer_id) {

  // ===== INPUT VALIDATION =====

  // Null pointer checks
  if (gradient == nullptr || g_current == nullptr || g_previous == nullptr ||
      output == nullptr) {
    return cudaErrorInvalidValue;
  }

  // Sanity check on N
  if (N <= 0) {
    return cudaErrorInvalidValue;
  }

  if (N > 1000000000) { // 1B elements = 4GB, reasonable upper bound
    return cudaErrorInvalidValue;
  }

  // Validate gamma (prevent division by zero and overflow)
  if (gamma < 1e-8f || gamma > 1e6f || isnan(gamma) || isinf(gamma)) {
    return cudaErrorInvalidValue;
  }

  // Validate momentum
  if (momentum < 0.0f || momentum > 1.0f || isnan(momentum) ||
      isinf(momentum)) {
    return cudaErrorInvalidValue;
  }

  // ===== KERNEL LAUNCH =====

  // Check alignment for vectorized path:
  // float4 requires 16-byte alignment, output requires 4-byte alignment for
  // uint32_t
  bool aligned =
      (N % 8 == 0) && (N >= 256) && (((uintptr_t)gradient % 16) == 0) &&
      (((uintptr_t)g_current % 16) == 0) &&
      (((uintptr_t)g_previous % 16) == 0) && (((uintptr_t)output % 4) == 0);

  if (aligned) {
    // Vectorized path (8 elements per thread)
    int N4 = N / 4;
    int threads = BLOCK_SIZE;
    int blocks = (N4 / 2 + threads - 1) / threads;

    predict_subtract_quantize_vec4_kernel<<<blocks, threads, 0, stream>>>(
        (const float4 *)gradient, (const float4 *)g_current,
        (const float4 *)g_previous, momentum, gamma, (uint32_t *)output, N4,
        step, layer_id);
  } else {
    // Fallback to scalar kernel for unaligned or small tensors
    int threads = BLOCK_SIZE;
    int elements_per_thread = 2;
    int blocks = (N + threads * elements_per_thread - 1) /
                 (threads * elements_per_thread);

    predict_subtract_quantize_kernel<<<blocks, threads, 0, stream>>>(
        gradient, g_current, g_previous, momentum, gamma, output, N, step,
        layer_id);
  }

  // ===== ERROR CHECKING =====

  // Check for kernel launch errors (synchronous)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }

  // Determine if we should synchronize based on sync_mode
  bool should_sync = false;

  switch (sync_mode) {
  case SYNC_ASYNC:
    // Never sync - maximum performance
    should_sync = false;
    break;

  case SYNC_ALWAYS:
    // Always sync - catch all errors immediately
    should_sync = true;
    break;

  case SYNC_PERIODIC:
    // Sync every 100 calls - balanced approach
    {
      // Race condition is acceptable for periodic error checking
      static int call_count = 0;
      call_count++;
      should_sync = (call_count % 100 == 0);
    }
    break;

  default:
    // Invalid sync mode, default to periodic
    should_sync = false;
    break;
  }

  // Synchronize if needed to catch kernel execution errors
  if (should_sync) {
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      return err;
    }
  }

  return cudaSuccess;
}

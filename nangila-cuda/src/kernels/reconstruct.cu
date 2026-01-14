/*
 * Nangila CUDA Kernels: Fused Dequantize-Add-Reconstruct
 *
 * This kernel fuses the decompression operations:
 * 1. Unpack: Extract INT4 values from packed bytes
 * 2. Dequantize: r = q * γ
 * 3. Predict: ĝ = g_t + μ * (g_t - g_{t-1})
 * 4. Add: g = ĝ + r
 *
 * Used on the receiver side to reconstruct gradients.
 */

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

// INT4 unpacking
__device__ __forceinline__ void unpack_int4(uint8_t packed, int8_t *a,
                                            int8_t *b) {
  int8_t low = packed & 0x0F;
  int8_t high = (packed >> 4) & 0x0F;
  // Sign extend from 4 bits
  *a = (low & 0x08) ? (low | 0xF0) : low;
  *b = (high & 0x08) ? (high | 0xF0) : high;
}

/*
 * Fused Dequantize-Add-Reconstruct Kernel
 *
 * Inputs:
 *   packed:      Packed INT4 quantized residuals [N/2]
 *   g_current:   Gradient from step t (for prediction) [N]
 *   g_previous:  Gradient from step t-1 (for prediction) [N]
 *   momentum:    Momentum coefficient (typically 0.9)
 *   gamma:       Quantization scale factor
 *   N:           Number of elements
 *
 * Outputs:
 *   output:      Reconstructed gradient [N]
 */
extern "C" __global__ void dequantize_add_reconstruct_kernel(
    const uint8_t *__restrict__ packed, const float *__restrict__ g_current,
    const float *__restrict__ g_previous, float momentum, float gamma,
    float *__restrict__ output, int N) {
  // Each thread handles 2 elements (from 1 packed byte)
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  if (idx >= N)
    return;

  // Unpack two INT4 values
  int8_t q0, q1;
  unpack_int4(packed[idx / 2], &q0, &q1);

  // Process first element
  float curr0 = g_current[idx];
  float prev0 = g_previous[idx];

  // Predict: ĝ = g_t + μ * (g_t - g_{t-1})
  float pred0 = curr0 + momentum * (curr0 - prev0);

  // Dequantize: r = q * γ
  float residual0 = (float)q0 * gamma;

  // Reconstruct: g = ĝ + r
  output[idx] = pred0 + residual0;

  // Process second element (if within bounds)
  if (idx + 1 < N) {
    float curr1 = g_current[idx + 1];
    float prev1 = g_previous[idx + 1];

    float pred1 = curr1 + momentum * (curr1 - prev1);
    float residual1 = (float)q1 * gamma;
    output[idx + 1] = pred1 + residual1;
  }
}

/*
 * Vectorized version for better memory coalescing
 */
extern "C" __global__ void dequantize_add_reconstruct_vec4_kernel(
    const uint32_t *__restrict__ packed, // 4 bytes = 8 INT4 values
    const float4 *__restrict__ g_current, const float4 *__restrict__ g_previous,
    float momentum, float gamma, float4 *__restrict__ output,
    int N4 // N / 4
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx * 2 >= N4)
    return;

  // Load packed data (4 bytes = 8 INT4 values)
  uint32_t pack = packed[idx];

  // Unpack 8 INT4 values
  int8_t q[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    int8_t nibble = (pack >> (i * 4)) & 0x0F;
    q[i] = (nibble & 0x08) ? (nibble | 0xF0) : nibble;
  }

  // Load current and previous gradients
  float4 curr_a = g_current[idx * 2];
  float4 curr_b = g_current[idx * 2 + 1];
  float4 prev_a = g_previous[idx * 2];
  float4 prev_b = g_previous[idx * 2 + 1];

  // Reconstruct
  float4 out_a, out_b;

#define RECONSTRUCT(out, curr, prev, qi)                                       \
  out = curr + momentum * (curr - prev) + (float)q[qi] * gamma

  RECONSTRUCT(out_a.x, curr_a.x, prev_a.x, 0);
  RECONSTRUCT(out_a.y, curr_a.y, prev_a.y, 1);
  RECONSTRUCT(out_a.z, curr_a.z, prev_a.z, 2);
  RECONSTRUCT(out_a.w, curr_a.w, prev_a.w, 3);
  RECONSTRUCT(out_b.x, curr_b.x, prev_b.x, 4);
  RECONSTRUCT(out_b.y, curr_b.y, prev_b.y, 5);
  RECONSTRUCT(out_b.z, curr_b.z, prev_b.z, 6);
  RECONSTRUCT(out_b.w, curr_b.w, prev_b.w, 7);

#undef RECONSTRUCT

  // Write output
  output[idx * 2] = out_a;
  output[idx * 2 + 1] = out_b;
}

/*
 * Host wrapper function
 */
extern "C" void launch_dequantize_add_reconstruct(
    const uint8_t *packed, const float *g_current, const float *g_previous,
    float momentum, float gamma, float *output, int N, cudaStream_t stream) {
  if (N % 8 == 0 && N >= 256) {
    int N4 = N / 4;
    int threads = BLOCK_SIZE;
    int blocks = (N4 / 2 + threads - 1) / threads;

    dequantize_add_reconstruct_vec4_kernel<<<blocks, threads, 0, stream>>>(
        (const uint32_t *)packed, (const float4 *)g_current,
        (const float4 *)g_previous, momentum, gamma, (float4 *)output, N4);
  } else {
    int threads = BLOCK_SIZE;
    int elements_per_thread = 2;
    int blocks = (N + threads * elements_per_thread - 1) /
                 (threads * elements_per_thread);

    dequantize_add_reconstruct_kernel<<<blocks, threads, 0, stream>>>(
        packed, g_current, g_previous, momentum, gamma, output, N);
  }
}

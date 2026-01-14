/*
 * Nangila CUDA Kernels: Fused Predict-Subtract-Quantize
 *
 * This kernel fuses three operations into one memory-bandwidth-optimal pass:
 * 1. Predict: ĝ = g_t + μ * (g_t - g_{t-1})
 * 2. Subtract: r = g - ĝ
 * 3. Quantize: q = clamp(round(r / γ), -8, 7)
 *
 * Memory access pattern:
 * - Read: gradient (current), g_t, g_{t-1}
 * - Write: packed INT4 output
 *
 * All intermediate values stay in registers = minimal memory bandwidth.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Configuration
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// INT4 packing: two values per byte
// Low nibble = first value, high nibble = second value
__device__ __forceinline__ uint8_t pack_int4(int8_t a, int8_t b) {
    return ((uint8_t)(a & 0x0F)) | (((uint8_t)(b & 0x0F)) << 4);
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
 *
 * Outputs:
 *   output:      Packed INT4 quantized residuals [N/2]
 */
extern "C" __global__ void predict_subtract_quantize_kernel(
    const float* __restrict__ gradient,
    const float* __restrict__ g_current,
    const float* __restrict__ g_previous,
    float momentum,
    float gamma,
    uint8_t* __restrict__ output,
    int N
) {
    // Each thread handles 2 elements (packed into 1 byte)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    if (idx >= N) return;
    
    // Process first element
    float g0 = gradient[idx];
    float curr0 = g_current[idx];
    float prev0 = g_previous[idx];
    
    // Predict: ĝ = g_t + μ * (g_t - g_{t-1})
    float pred0 = curr0 + momentum * (curr0 - prev0);
    
    // Subtract: r = g - ĝ
    float residual0 = g0 - pred0;
    
    // Quantize: clamp(round(r / γ), -8, 7)
    float scaled0 = residual0 / gamma;
    int8_t quant0 = (int8_t)max(-8, min(7, (int)roundf(scaled0)));
    
    // Process second element (if within bounds)
    int8_t quant1 = 0;
    if (idx + 1 < N) {
        float g1 = gradient[idx + 1];
        float curr1 = g_current[idx + 1];
        float prev1 = g_previous[idx + 1];
        
        float pred1 = curr1 + momentum * (curr1 - prev1);
        float residual1 = g1 - pred1;
        float scaled1 = residual1 / gamma;
        quant1 = (int8_t)max(-8, min(7, (int)roundf(scaled1)));
    }
    
    // Pack two INT4 values into one byte
    output[idx / 2] = pack_int4(quant0, quant1);
}

/*
 * Vectorized version using float4 for better memory coalescing
 * Processes 8 elements per thread (4 output bytes)
 */
extern "C" __global__ void predict_subtract_quantize_vec4_kernel(
    const float4* __restrict__ gradient,
    const float4* __restrict__ g_current,
    const float4* __restrict__ g_previous,
    float momentum,
    float gamma,
    uint32_t* __restrict__ output,  // 4 bytes = 8 INT4 values
    int N4  // N / 4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * 2 >= N4) return;
    
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
    
    #define PROCESS(i, ga, ca, pa) do { \
        float pred = ca + momentum * (ca - pa); \
        float residual = ga - pred; \
        float scaled = residual * inv_gamma; \
        q[i] = (int8_t)max(-8, min(7, (int)roundf(scaled))); \
    } while(0)
    
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
 * Host wrapper function
 */
extern "C" void launch_predict_subtract_quantize(
    const float* gradient,
    const float* g_current,
    const float* g_previous,
    float momentum,
    float gamma,
    uint8_t* output,
    int N,
    cudaStream_t stream
) {
    // Use vectorized kernel if N is divisible by 8
    if (N % 8 == 0 && N >= 256) {
        int N4 = N / 4;
        int threads = BLOCK_SIZE;
        int blocks = (N4 / 2 + threads - 1) / threads;
        
        predict_subtract_quantize_vec4_kernel<<<blocks, threads, 0, stream>>>(
            (const float4*)gradient,
            (const float4*)g_current,
            (const float4*)g_previous,
            momentum,
            gamma,
            (uint32_t*)output,
            N4
        );
    } else {
        // Fallback to scalar kernel
        int threads = BLOCK_SIZE;
        int elements_per_thread = 2;
        int blocks = (N + threads * elements_per_thread - 1) / (threads * elements_per_thread);
        
        predict_subtract_quantize_kernel<<<blocks, threads, 0, stream>>>(
            gradient,
            g_current,
            g_previous,
            momentum,
            gamma,
            output,
            N
        );
    }
}

/*
 * Nangila CUDA Kernels: Dynamic Gamma Computation
 *
 * Computes the quantization scale (gamma) based on the 99th percentile
 * of absolute residual values. This ensures most values fit within
 * the INT4 range [-8, 7] without clipping.
 */

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

/*
 * Reduction kernel to find max absolute value (for initial gamma estimate)
 */
extern "C" __global__ void find_max_abs_kernel(const float *__restrict__ data,
                                               float *__restrict__ block_maxes,
                                               int N) {
  typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float local_max = 0.0f;
  if (idx < N) {
    local_max = fabsf(data[idx]);
  }

  float block_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());

  if (threadIdx.x == 0) {
    block_maxes[blockIdx.x] = block_max;
  }
}

/*
 * Histogram kernel for percentile computation
 * Bins values into 256 buckets for approximate percentile
 */
extern "C" __global__ void
histogram_abs_kernel(const float *__restrict__ data,
                     unsigned int *__restrict__ histogram, float max_val,
                     int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  float val = fabsf(data[idx]);
  float normalized = val / max_val; // 0 to 1
  int bin = min(255, (int)(normalized * 256.0f));

  atomicAdd(&histogram[bin], 1);
}

/*
 * Compute gamma from histogram (find 99th percentile)
 * This runs on CPU after histogram is computed
 */
extern "C" float compute_gamma_from_histogram(unsigned int *histogram,
                                              float max_val, int N,
                                              float percentile // e.g., 0.99
) {
  int target_count = (int)(N * percentile);
  int cumulative = 0;
  int p99_bin = 255;

  for (int i = 0; i < 256; i++) {
    cumulative += histogram[i];
    if (cumulative >= target_count) {
      p99_bin = i;
      break;
    }
  }

  float p99_val = ((float)p99_bin / 256.0f) * max_val;

  // gamma = p99 / 7 (INT4 max positive value)
  return fmaxf(p99_val / 7.0f, 1e-8f);
}

/*
 * Simple gamma computation using max absolute value
 * Faster but less robust to outliers
 */
extern "C" __global__ void
compute_gamma_simple_kernel(const float *__restrict__ block_maxes,
                            float *__restrict__ gamma_out, int num_blocks) {
  // Single-block reduction
  typedef cub::BlockReduce<float, 1024> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  float local_max = 0.0f;
  for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
    local_max = fmaxf(local_max, block_maxes[i]);
  }

  float global_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());

  if (threadIdx.x == 0) {
    // gamma = max / 7, with minimum to prevent division by zero
    *gamma_out = fmaxf(global_max / 7.0f, 1e-8f);
  }
}

/*
 * Host wrapper for gamma computation
 */
extern "C" void
launch_compute_gamma(const float *residuals, float *gamma_out,
                     float *workspace, // Size: num_blocks * sizeof(float)
                     int N, cudaStream_t stream) {
  int threads = BLOCK_SIZE;
  int blocks = (N + threads - 1) / threads;

  // Step 1: Find max absolute value per block
  find_max_abs_kernel<<<blocks, threads, 0, stream>>>(residuals, workspace, N);

  // Step 2: Reduce block maxes to single gamma
  compute_gamma_simple_kernel<<<1, 1024, 0, stream>>>(workspace, gamma_out,
                                                      blocks);
}

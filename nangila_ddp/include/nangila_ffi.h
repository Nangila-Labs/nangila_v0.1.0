/*
 * Nangila FFI Header
 *
 * Declares extern "C" functions exported by libnangila_hook
 * for use in the C++ DDP hook implementation.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to Rust NangilaHook
typedef void *NangilaHandle;

// Data type constants (match Rust DataType enum)
#define NANGILA_DTYPE_FLOAT32 0
#define NANGILA_DTYPE_FLOAT16 1
#define NANGILA_DTYPE_BFLOAT16 2

// Error codes
#define NANGILA_SUCCESS 0
#define NANGILA_INVALID_POINTER 1
#define NANGILA_INVALID_PATH 2
#define NANGILA_IO_ERROR 3
#define NANGILA_COMPRESSION_ERROR 4
#define NANGILA_DECOMPRESSION_ERROR 5

// Initialize with all-driver topology (no Passengers)
NangilaHandle nangila_init_all_drivers(uint32_t num_layers);

// Initialize from mask file
NangilaHandle nangila_init(const char *mask_path);

// Free handle
void nangila_free(NangilaHandle handle);

// Compress gradient (FP32)
int32_t nangila_compress(NangilaHandle handle, const float *sendbuff,
                         size_t count, uint32_t layer_id,
                         uint8_t *out_compressed, size_t *out_compressed_size);

// Decompress gradient (FP32)
int32_t nangila_decompress(NangilaHandle handle, const uint8_t *compressed,
                           size_t compressed_size, uint32_t layer_id,
                           float *recvbuff, size_t count);

// Compress with explicit dtype (FP16/BF16 support)
int32_t nangila_compress_ex(NangilaHandle handle, const uint8_t *sendbuff,
                            size_t count, int32_t dtype, uint32_t layer_id,
                            uint8_t *out_compressed,
                            size_t *out_compressed_size);

// Decompress with explicit dtype
int32_t nangila_decompress_ex(NangilaHandle handle, const uint8_t *compressed,
                              size_t compressed_size, uint32_t layer_id,
                              uint8_t *recvbuff, size_t count, int32_t dtype);

// Update predictor state after All-Reduce
int32_t nangila_on_complete(NangilaHandle handle, uint32_t layer_id,
                            const float *gradient, size_t count);

// Advance to next training step
int32_t nangila_step(NangilaHandle handle);

// Check if compression is enabled
int32_t nangila_is_enabled(NangilaHandle handle);

// Get current step number
uint64_t nangila_current_step(NangilaHandle handle);

// Get maximum compressed buffer size
size_t nangila_max_compressed_size(size_t count);

#ifdef __cplusplus
}
#endif

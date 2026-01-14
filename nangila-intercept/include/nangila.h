/*
 * Nangila FFI Header
 *
 * C interface for Nangila gradient compression library.
 * This header is used by the NCCL intercept shim.
 */

#ifndef NANGILA_H
#define NANGILA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define NANGILA_SUCCESS 0
#define NANGILA_INVALID_POINTER 1
#define NANGILA_INVALID_PATH 2
#define NANGILA_IO_ERROR 3
#define NANGILA_COMPRESSION_ERROR 4
#define NANGILA_DECOMPRESSION_ERROR 5

/* Opaque handle to Nangila state */
typedef void *NangilaHandle;

/**
 * Initialize Nangila with a topology mask file.
 *
 * @param mask_path Path to .nzmask file (null-terminated string)
 * @return Handle to Nangila state, or NULL on failure
 */
NangilaHandle nangila_init(const char *mask_path);

/**
 * Initialize Nangila with all-driver topology (no passengers).
 * Useful for testing or when topology is unknown.
 *
 * @param num_layers Number of layers to track
 * @return Handle to Nangila state
 */
NangilaHandle nangila_init_all_drivers(uint32_t num_layers);

/**
 * Compress gradient data before sending over network.
 *
 * @param handle Nangila handle from nangila_init()
 * @param sendbuff Input gradient buffer (FP32)
 * @param count Number of elements in sendbuff
 * @param layer_id Layer identifier
 * @param out_compressed Output buffer for compressed data
 * @param out_compressed_size [in/out] Max size on input, actual size on output
 * @return 0 on success, error code on failure
 */
int32_t nangila_compress(NangilaHandle handle, const float *sendbuff,
                         size_t count, uint32_t layer_id,
                         uint8_t *out_compressed, size_t *out_compressed_size);

/**
 * Decompress received gradient data.
 *
 * @param handle Nangila handle
 * @param compressed Compressed data received from network
 * @param compressed_size Size of compressed data
 * @param layer_id Layer identifier
 * @param recvbuff Output buffer for decompressed gradient (FP32)
 * @param count Expected number of elements
 * @return 0 on success, error code on failure
 */
int32_t nangila_decompress(NangilaHandle handle, const uint8_t *compressed,
                           size_t compressed_size, uint32_t layer_id,
                           float *recvbuff, size_t count);

/**
 * Update state after All-Reduce completes.
 * Must be called after each layer's All-Reduce to update predictor.
 *
 * @param handle Nangila handle
 * @param layer_id Layer identifier
 * @param gradient Final gradient after All-Reduce
 * @param count Number of elements
 * @return 0 on success, error code on failure
 */
int32_t nangila_on_complete(NangilaHandle handle, uint32_t layer_id,
                            const float *gradient, size_t count);

/**
 * Advance to the next training step.
 * Must be called at the end of each training iteration.
 *
 * @param handle Nangila handle
 * @return 0 on success, error code on failure
 */
int32_t nangila_step(NangilaHandle handle);

/**
 * Check if compression is currently enabled.
 * Compression is disabled during warmup period.
 *
 * @param handle Nangila handle
 * @return 1 if enabled, 0 if disabled or invalid handle
 */
int32_t nangila_is_enabled(NangilaHandle handle);

/**
 * Get current training step number.
 *
 * @param handle Nangila handle
 * @return Current step, or 0 on invalid handle
 */
uint64_t nangila_current_step(NangilaHandle handle);

/**
 * Get predictor state hash for verification.
 * Can be used to verify synchronization between nodes.
 *
 * @param handle Nangila handle
 * @return 64-bit hash of predictor state
 */
uint64_t nangila_predictor_hash(NangilaHandle handle);

/**
 * Free Nangila handle.
 * Must be called exactly once for each successful nangila_init() call.
 *
 * @param handle Nangila handle to free
 */
void nangila_free(NangilaHandle handle);

/**
 * Get maximum compressed buffer size.
 * Use this to allocate output buffer for nangila_compress().
 *
 * @param count Number of elements in uncompressed gradient
 * @return Maximum size needed for compressed output buffer
 */
size_t nangila_max_compressed_size(size_t count);

/* ==========================================================================
 * Extended API with Data Type Support (FP16/BF16)
 * ========================================================================== */

/* Data type constants */
#define NANGILA_DTYPE_FLOAT32 0
#define NANGILA_DTYPE_FLOAT16 1
#define NANGILA_DTYPE_BFLOAT16 2

/**
 * Compress gradient data with explicit data type.
 * Supports FP32, FP16, and BF16 gradients.
 *
 * @param handle Nangila handle from nangila_init()
 * @param sendbuff Input gradient buffer (raw bytes, any dtype)
 * @param count Number of elements in sendbuff
 * @param dtype Data type: NANGILA_DTYPE_FLOAT32/FLOAT16/BFLOAT16
 * @param layer_id Layer identifier
 * @param out_compressed Output buffer for compressed data
 * @param out_compressed_size [in/out] Max size on input, actual size on output
 * @return 0 on success, error code on failure
 */
int32_t nangila_compress_ex(NangilaHandle handle, const uint8_t *sendbuff,
                            size_t count, int32_t dtype, uint32_t layer_id,
                            uint8_t *out_compressed,
                            size_t *out_compressed_size);

/**
 * Decompress received gradient data with explicit data type.
 * Output is converted to the specified dtype.
 *
 * @param handle Nangila handle
 * @param compressed Compressed data received from network
 * @param compressed_size Size of compressed data
 * @param layer_id Layer identifier
 * @param recvbuff Output buffer for decompressed gradient (in specified dtype)
 * @param count Expected number of elements
 * @param dtype Data type: NANGILA_DTYPE_FLOAT32/FLOAT16/BFLOAT16
 * @return 0 on success, error code on failure
 */
int32_t nangila_decompress_ex(NangilaHandle handle, const uint8_t *compressed,
                              size_t compressed_size, uint32_t layer_id,
                              uint8_t *recvbuff, size_t count, int32_t dtype);

#ifdef __cplusplus
}
#endif

#endif /* NANGILA_H */

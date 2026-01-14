/*
 * Nangila NCCL Intercept Shim
 *
 * This library intercepts NCCL collective operations via LD_PRELOAD
 * and redirects them through Nangila gradient compression.
 *
 * Usage:
 *   LD_PRELOAD=./libnangila_intercept.so \
 *   NANGILA_MASK=topology.nzmask \
 *   python train.py
 *
 * Environment Variables:
 *   NANGILA_MASK      - Path to topology mask file (optional)
 *   NANGILA_NUM_LAYERS - Number of layers if no mask (default: 1000)
 *   NANGILA_DEBUG     - Enable debug logging (0 or 1)
 */

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/nangila.h"

/* NCCL types (from nccl.h) */
typedef void *ncclComm_t;
typedef int ncclResult_t;

typedef enum {
  ncclFloat = 0,
  ncclFloat32 = 0,
  ncclHalf = 1,
  ncclFloat16 = 1,
  ncclBfloat16 = 2,
  ncclDouble = 3,
  ncclFloat64 = 3,
  ncclInt32 = 4,
  ncclInt = 4,
  ncclInt64 = 5,
  ncclUint8 = 6,
  ncclChar = 6,
  ncclUint32 = 7,
  ncclUint64 = 8
} ncclDataType_t;

typedef enum {
  ncclSum = 0,
  ncclProd = 1,
  ncclMax = 2,
  ncclMin = 3,
  ncclAvg = 4
} ncclRedOp_t;

#define ncclSuccess 0

/* Function pointer for real NCCL functions */
typedef ncclResult_t (*ncclAllReduce_fn)(const void *, void *, size_t,
                                         ncclDataType_t, ncclRedOp_t,
                                         ncclComm_t, cudaStream_t);

/* Global state */
static NangilaHandle g_nangila = NULL;
static ncclAllReduce_fn g_real_ncclAllReduce = NULL;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_initialized = 0;
static int g_debug = 0;
static uint32_t g_layer_counter = 0;

/* Debug logging */
#define DEBUG_LOG(...)                                                         \
  do {                                                                         \
    if (g_debug)                                                               \
      fprintf(stderr, "[nangila] " __VA_ARGS__);                               \
  } while (0)

/* Initialize Nangila on first call */
static void init_nangila(void) {
  if (g_initialized)
    return;

  pthread_mutex_lock(&g_mutex);
  if (g_initialized) {
    pthread_mutex_unlock(&g_mutex);
    return;
  }

  /* Check debug flag */
  const char *debug_env = getenv("NANGILA_DEBUG");
  g_debug = (debug_env && strcmp(debug_env, "1") == 0);

  DEBUG_LOG("Initializing Nangila intercept shim\n");

  /* Get real ncclAllReduce */
  g_real_ncclAllReduce = (ncclAllReduce_fn)dlsym(RTLD_NEXT, "ncclAllReduce");
  if (!g_real_ncclAllReduce) {
    fprintf(stderr, "[nangila] ERROR: Could not find real ncclAllReduce: %s\n",
            dlerror());
    pthread_mutex_unlock(&g_mutex);
    return;
  }
  DEBUG_LOG("Found real ncclAllReduce at %p\n", (void *)g_real_ncclAllReduce);

  /* Initialize Nangila */
  const char *mask_path = getenv("NANGILA_MASK");
  if (mask_path && strlen(mask_path) > 0) {
    DEBUG_LOG("Loading mask from %s\n", mask_path);
    g_nangila = nangila_init(mask_path);
  } else {
    const char *num_layers_env = getenv("NANGILA_NUM_LAYERS");
    uint32_t num_layers =
        num_layers_env ? (uint32_t)atoi(num_layers_env) : 1000;
    DEBUG_LOG("Initializing with %u all-driver layers\n", num_layers);
    g_nangila = nangila_init_all_drivers(num_layers);
  }

  if (!g_nangila) {
    fprintf(stderr, "[nangila] ERROR: Failed to initialize Nangila\n");
    pthread_mutex_unlock(&g_mutex);
    return;
  }

  DEBUG_LOG("Nangila initialized successfully\n");
  g_initialized = 1;
  pthread_mutex_unlock(&g_mutex);
}

/* Cleanup on library unload */
__attribute__((destructor)) static void cleanup_nangila(void) {
  if (g_nangila) {
    DEBUG_LOG("Cleaning up Nangila\n");
    nangila_free(g_nangila);
    g_nangila = NULL;
  }
}

/* Get element size for NCCL datatype */
static size_t get_dtype_size(ncclDataType_t dtype) {
  switch (dtype) {
  case ncclFloat32:
    return 4;
  case ncclFloat16:
    return 2;
  case ncclBfloat16:
    return 2;
  case ncclFloat64:
    return 8;
  case ncclInt32:
    return 4;
  case ncclInt64:
    return 8;
  case ncclUint8:
    return 1;
  case ncclUint32:
    return 4;
  case ncclUint64:
    return 8;
  default:
    return 4;
  }
}

/* Convert NCCL dtype to Nangila dtype */
static int32_t nccl_to_nangila_dtype(ncclDataType_t dtype) {
  switch (dtype) {
  case ncclFloat32:
    return NANGILA_DTYPE_FLOAT32;
  case ncclFloat16:
    return NANGILA_DTYPE_FLOAT16;
  case ncclBfloat16:
    return NANGILA_DTYPE_BFLOAT16;
  default:
    return -1; /* Unsupported */
  }
}

/* Intercepted ncclAllReduce */
ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  init_nangila();

  /* Get Nangila dtype, fall back if unsupported */
  int32_t nangila_dtype = nccl_to_nangila_dtype(datatype);

  /* Fall back to real NCCL if not initialized or unsupported dtype */
  if (!g_nangila || !g_real_ncclAllReduce || nangila_dtype < 0) {
    if (g_real_ncclAllReduce) {
      return g_real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                                  stream);
    }
    return ncclSuccess; /* Should not happen */
  }

  /* Get layer ID (simple counter, could be improved with tensor address
   * hashing) */
  uint32_t layer_id = __sync_fetch_and_add(&g_layer_counter, 1);
  size_t elem_size = get_dtype_size(datatype);

  DEBUG_LOG("AllReduce layer %u: count=%zu dtype=%d\n", layer_id, count,
            nangila_dtype);

  if (!nangila_is_enabled(g_nangila)) {
    /* During warmup, just pass through */
    DEBUG_LOG("  Warmup mode, passing through\n");
    ncclResult_t result = g_real_ncclAllReduce(sendbuff, recvbuff, count,
                                               datatype, op, comm, stream);

    /* Still need to update predictor state (convert to FP32 for predictor) */
    cudaStreamSynchronize(stream);

    /* For warmup, only track FP32 for simplicity */
    if (datatype == ncclFloat32) {
      nangila_on_complete(g_nangila, layer_id, (const float *)recvbuff, count);
    }

    return result;
  }

  /* Allocate host buffers */
  size_t host_size = count * elem_size;
  size_t compressed_size = nangila_max_compressed_size(count);

  uint8_t *h_send = (uint8_t *)malloc(host_size);
  uint8_t *h_recv = (uint8_t *)malloc(host_size);
  uint8_t *h_compressed = (uint8_t *)malloc(compressed_size);

  if (!h_send || !h_recv || !h_compressed) {
    fprintf(stderr, "[nangila] ERROR: Memory allocation failed\n");
    free(h_send);
    free(h_recv);
    free(h_compressed);
    return g_real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                                stream);
  }

  /* Copy gradient to host */
  cudaMemcpyAsync(h_send, sendbuff, host_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  /* Compress with dtype support */
  size_t actual_compressed_size = compressed_size;
  int32_t result =
      nangila_compress_ex(g_nangila, h_send, count, nangila_dtype, layer_id,
                          h_compressed, &actual_compressed_size);

  if (result != NANGILA_SUCCESS) {
    DEBUG_LOG("  Compression failed (%d), falling back\n", result);
    free(h_send);
    free(h_recv);
    free(h_compressed);
    return g_real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                                stream);
  }

  DEBUG_LOG("  Compressed %zu -> %zu bytes (%.1fx)\n", host_size,
            actual_compressed_size, (float)host_size / actual_compressed_size);

  /*
   * TODO: For real implementation, we would:
   * 1. Allocate device buffer for compressed data
   * 2. Copy compressed to device
   * 3. Call real ncclAllReduce with compressed size
   * 4. Copy back and decompress
   *
   * For now, we simulate by just doing the original AllReduce
   * (the compression/decompression overhead is still measured)
   */
  ncclResult_t nccl_result = g_real_ncclAllReduce(sendbuff, recvbuff, count,
                                                  datatype, op, comm, stream);
  cudaStreamSynchronize(stream);

  /* Copy result to host */
  cudaMemcpyAsync(h_recv, recvbuff, host_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  /* Decompress with dtype support (simulated - we actually use the real result)
   */
  /* In production, this would decompress h_compressed into h_recv */

  /* Update predictor state (using FP32 internally) */
  if (datatype == ncclFloat32) {
    nangila_on_complete(g_nangila, layer_id, (const float *)h_recv, count);
  }
  /* TODO: Add nangila_on_complete_ex for FP16/BF16 predictor updates */

  free(h_send);
  free(h_recv);
  free(h_compressed);

  return nccl_result;
}

/* Also intercept ncclBroadcast, ncclReduce, etc. as needed */
/* For now, we only handle AllReduce which is the main DDP bottleneck */

/* Step counter - called from Python via ctypes if needed */
__attribute__((visibility("default"))) void nangila_intercept_step(void) {
  if (g_nangila) {
    nangila_step(g_nangila);
    g_layer_counter = 0; /* Reset layer counter for next iteration */
    DEBUG_LOG("Step complete, layer counter reset\n");
  }
}

/* Get compression stats */
__attribute__((visibility("default"))) uint64_t
nangila_intercept_get_step(void) {
  return g_nangila ? nangila_current_step(g_nangila) : 0;
}

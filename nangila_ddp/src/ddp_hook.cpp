/*
 * Nangila DDP Communication Hook (C++ Implementation)
 *
 * This module provides a native C++ DDP hook that bypasses Python's
 * serialization overhead by directly integrating with PyTorch's C++ API.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/torch.h>

#include "nangila_ffi.h"

#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace nangila {

// Convert torch dtype to Nangila dtype constant
inline int32_t torch_to_nangila_dtype(c10::ScalarType dtype) {
  switch (dtype) {
  case c10::ScalarType::Float:
    return NANGILA_DTYPE_FLOAT32;
  case c10::ScalarType::Half:
    return NANGILA_DTYPE_FLOAT16;
  case c10::ScalarType::BFloat16:
    return NANGILA_DTYPE_BFLOAT16;
  default:
    return NANGILA_DTYPE_FLOAT32;
  }
}

/**
 * NangilaDDPHook - Native C++ DDP Communication Hook
 *
 * This hook compresses gradients before All-Reduce and decompresses after,
 * reducing network bandwidth by ~30x while maintaining training accuracy.
 */
class NangilaDDPHook {
public:
  NangilaDDPHook(int num_layers, int warmup_steps = 0)
      : num_layers_(num_layers), warmup_steps_(warmup_steps), current_step_(0),
        layer_counter_(0) {

    // Initialize Rust hook with fast-forward for warmup
    rust_hook_ = nangila_init_all_drivers(static_cast<uint32_t>(num_layers));
    if (!rust_hook_) {
      throw std::runtime_error("Failed to initialize Nangila hook");
    }

    // Enable GPU/Stateless mode for efficiency
    nangila_set_gpu_mode(rust_hook_, 1);

    // Fast-forward past warmup
    for (int i = 0; i < warmup_steps + 100; ++i) {
      nangila_step(rust_hook_);
    }
  }

  ~NangilaDDPHook() {
    if (rust_hook_) {
      nangila_free(rust_hook_);
    }
  }

  // Non-copyable
  NangilaDDPHook(const NangilaDDPHook &) = delete;
  NangilaDDPHook &operator=(const NangilaDDPHook &) = delete;

  /**
   * Hook operator - GPU Native
   */
  /**
   * Enable Safe Mode
   */
  void enable_safe_mode(float divergence_threshold, int check_interval,
                        int max_failures, int recovery_cooldown) {
    if (rust_hook_) {
      nangila_enable_safe_mode(rust_hook_, divergence_threshold, check_interval,
                               static_cast<uint32_t>(max_failures),
                               recovery_cooldown);
    }
  }

  /**
   * Report validation loss
   */
  int report_validation_loss(float loss) {
    if (rust_hook_) {
      return nangila_report_val_loss(rust_hook_, loss);
    }
    return 3; // NoCheck
  }

  /**
   * Hook operator - GPU Native
   */
  c10::intrusive_ptr<c10::ivalue::Future> run_hook(py::object pg_obj,
                                                   py::object bucket_obj) {
    fprintf(stderr, "[C++] run_hook called\n");
    // Manually cast arguments to avoid signature mismatch issues
    // with PyTorch's ProcessGroup bindings
    c10::intrusive_ptr<c10d::ProcessGroup> pg;
    try {
      fprintf(stderr, "[C++] Casting pg_obj...\n");
      pg = pg_obj.cast<c10::intrusive_ptr<c10d::ProcessGroup>>();
      fprintf(stderr, "[C++] pg_obj cast success\n");
    } catch (const std::exception &e) {
      fprintf(stderr, "[C++] Failed to cast ProcessGroup: %s\n", e.what());
      throw std::runtime_error("Failed to cast ProcessGroup: " +
                               std::string(e.what()));
    }

    torch::Tensor bucket_tensor;
    try {
      fprintf(stderr, "[C++] Casting bucket_obj...\n");
      bucket_tensor = bucket_obj.cast<torch::Tensor>();
      fprintf(stderr, "[C++] bucket_obj cast success\n");
    } catch (const std::exception &e) {
      fprintf(stderr, "[C++] Failed to cast bucket Tensor: %s\n", e.what());
      throw std::runtime_error("Failed to cast bucket Tensor: " +
                               std::string(e.what()));
    }

    // Get layer ID
    fprintf(stderr, "[C++] Fetching layer ID...\n");
    uint32_t layer_id = layer_counter_.fetch_add(1);
    fprintf(stderr, "[C++] Layer ID: %u\n", layer_id);

    // Ensure input is on GPU and valid
    if (!bucket_tensor.is_cuda()) {
      throw std::runtime_error(
          "Nangila GPU-Native pipeline requires CUDA tensors");
    }

    // Get current stream
    fprintf(stderr, "[C++] Getting CUDA stream...\n");
    c10::cuda::CUDAStream current_stream = c10::cuda::getCurrentCUDAStream();
    void *stream_ptr = current_stream.stream();
    fprintf(stderr, "[C++] Stream obtained: %p\n", stream_ptr);

    // 1. Get History (init on device if needed)
    fprintf(stderr, "[C++] Locking mutex...\n");
    std::lock_guard<std::mutex> lock(history_mutex_);

    fprintf(stderr, "[C++] Accessing history map...\n");
    auto &layer_hist = history_[layer_id];

    // Use bucket numel
    int64_t numel = bucket_tensor.numel();

    // Check definition AND size match
    if (!layer_hist.first.defined() || layer_hist.first.numel() != numel) {
      fprintf(stderr, "[C++] Initializing history tensors (Reset/Init)...\n");
      // Use explicit options
      auto opts = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(torch::kCUDA, current_stream.device_index());

      fprintf(stderr, "[C++] Creating history with numel=%ld\n", numel);

      // Ensure we clear old tensors if any
      layer_hist.first = torch::zeros({numel}, opts);
      layer_hist.second = torch::zeros({numel}, opts);
      fprintf(stderr, "[C++] History initialized.\n");
    }
    torch::Tensor g_prev = layer_hist.first;
    torch::Tensor g_curr = layer_hist.second;

    // 2. Prepare Compression Buffer
    const size_t max_compressed = nangila_max_compressed_size(numel);

    // Re-use scratch buffer if possible or alloc new
    fprintf(stderr, "[C++] Allocating compressed buffer...\n");
    // Explicit options again
    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::kCUDA, current_stream.device_index());

    torch::Tensor compressed_gpu =
        torch::empty({static_cast<int64_t>(max_compressed)}, options);
    fprintf(stderr, "[C++] Buffer allocated\n");

    // 3. Compress on GPU
    std::vector<uint8_t> header_cpu(24);
    size_t actual_size = 0;

    // Pointers
    fprintf(stderr, "[C++] Getting grad_ptr...\n");
    const float *grad_ptr = bucket_tensor.data_ptr<float>();

    fprintf(stderr, "[C++] Getting curr_ptr...\n");
    const float *curr_ptr = g_curr.data_ptr<float>();

    fprintf(stderr, "[C++] Getting prev_ptr...\n");
    const float *prev_ptr = g_prev.data_ptr<float>();

    fprintf(stderr, "[C++] Getting out_ptr...\n");
    uint8_t *out_ptr = compressed_gpu.data_ptr<uint8_t>();

    fprintf(stderr, "[C++] Pointers obtained\n");

    // Allocate scratch for CRC32
    auto scratch_opts =
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA, current_stream.device_index());
    torch::Tensor scratch = torch::empty({1}, scratch_opts);
    uint32_t *scratch_ptr =
        reinterpret_cast<uint32_t *>(scratch.data_ptr<int32_t>());

    // Offset payload by 24 bytes for header
    uint8_t *payload_ptr = out_ptr + 24;

    fprintf(stderr, "[C++] Calling nangila_compress_gpu...\n");
    int result = nangila_compress_gpu(
        rust_hook_, grad_ptr, curr_ptr, prev_ptr, numel, layer_id, payload_ptr,
        &actual_size, header_cpu.data(), stream_ptr, scratch_ptr);
    fprintf(stderr, "[C++] nangila_compress_gpu returned: %d\n", result);

    if (result != NANGILA_SUCCESS) {
      // Fallback or error
      return fallback_allreduce(pg, bucket_tensor);
    }

    // 4. Copy Header to GPU (Sync to ensure safety)
    AT_CUDA_CHECK(
        cudaMemcpy(out_ptr, header_cpu.data(), 24, cudaMemcpyHostToDevice));

    // 5. Size Handshake (Sync logic relative to network)
    // We need to know max size across ranks to enable all-gather
    // This requires a CPU roundtrip for sizes usually, OR we use a fixed-size
    // all-gather of sizes

    int world_size = pg->getSize();

    // Create size tensor on GPU or CPU? ProcessGroup operates on Tensors.
    // Small tensors on GPU are fine.
    auto size_opts = torch::TensorOptions()
                         .dtype(torch::kLong)
                         .device(bucket_tensor.device());
    torch::Tensor my_size =
        torch::tensor({static_cast<int64_t>(actual_size)}, size_opts);

    // Gather sizes
    std::vector<std::vector<torch::Tensor>> gathered_sizes(world_size);
    for (int i = 0; i < world_size; ++i) {
      gathered_sizes[i].resize(1);
      // ALLOCATE output tensor!
      gathered_sizes[i][0] = torch::empty_like(my_size);
    }
    std::vector<torch::Tensor> input_sizes = {my_size};

    auto size_work = pg->allgather(gathered_sizes, input_sizes);
    size_work->wait();

    // Compute max size
    int64_t max_size = 0;
    std::vector<int64_t> actual_sizes(world_size);

    for (int i = 0; i < world_size; ++i) {
      // work->wait() synchronizes CPU with the completion of comms, but data is
      // on GPU. We need to read it.
      int64_t s = gathered_sizes[i][0].item<int64_t>(); // Implicit sync
      actual_sizes[i] = s;
      if (s > max_size)
        max_size = s;
    }

    // 6. All-Gather Data
    // Slice our buffer to max_size (padding with garbage if actual < max)
    torch::Tensor send_tensor = compressed_gpu.slice(0, 0, max_size);

    std::vector<std::vector<torch::Tensor>> gathered_data(world_size);
    for (int i = 0; i < world_size; ++i) {
      gathered_data[i].resize(1);
      gathered_data[i][0] = torch::empty({max_size}, options);
    }
    std::vector<torch::Tensor> input_data = {send_tensor};

    auto data_work = pg->allgather(gathered_data, input_data);
    data_work->wait();

    // 7. Decompress and Accumulate
    torch::Tensor accum = torch::zeros_like(bucket_tensor);
    torch::Tensor temp = torch::empty_like(bucket_tensor);

    for (int i = 0; i < world_size; ++i) {
      torch::Tensor packet = gathered_data[i][0]; // GPU
      size_t packet_size = actual_sizes[i];

      // Decompress
      result = nangila_decompress_gpu(
          rust_hook_, packet.data_ptr<uint8_t>(), packet_size,
          curr_ptr, // Global History
          prev_ptr, layer_id, temp.data_ptr<float>(), numel, stream_ptr,
          scratch_ptr);

      if (result != NANGILA_SUCCESS) {
        // REPORT CRITICAL SAFETY ERROR
        fprintf(stderr,
                "[C++] nangila_decompress_gpu FAILED (Rank %d): Error %d\n", i,
                result);
        throw std::runtime_error(
            "Nangila Safety Violation: Decompression failed/corrupted packet "
            "(Error " +
            std::to_string(result) + ")");
      }

      accum.add_(temp);
    }

    // 8. Average
    accum.div_(world_size);

    // 9. Update History
    // New Prev = Old Curr
    // New Curr = New Global Average (accum)
    layer_hist.first.copy_(layer_hist.second);
    layer_hist.second.copy_(accum);

    // 10. Copy result to bucket
    bucket_tensor.copy_(accum);

    auto future =
        c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
    future->markCompleted(bucket_tensor);
    return future;
  }

  void step() {
    nangila_step(rust_hook_);
    layer_counter_.store(0);
    ++current_step_;
  }

  bool is_compression_enabled() const {
    return nangila_is_enabled(rust_hook_) != 0;
  }

  int64_t current_step() const { return current_step_; }

private:
  NangilaHandle rust_hook_;
  int num_layers_;
  int warmup_steps_;
  int64_t current_step_;
  std::atomic<uint32_t> layer_counter_;

  // History: Map<layer_id, Pair<Prev, Curr>>
  std::mutex history_mutex_;
  std::map<uint32_t, std::pair<torch::Tensor, torch::Tensor>> history_;

  c10::intrusive_ptr<c10::ivalue::Future>
  fallback_allreduce(c10::intrusive_ptr<c10d::ProcessGroup> pg,
                     torch::Tensor &bucket_tensor) {
    std::vector<torch::Tensor> tensors = {bucket_tensor};
    auto work = pg->allreduce(tensors);
    auto future =
        c10::make_intrusive<c10::ivalue::Future>(c10::TensorType::get());
    work->wait();
    bucket_tensor.div_(pg->getSize());
    future->markCompleted(bucket_tensor);
    return future;
  }
};

} // namespace nangila

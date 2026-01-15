/*
 * Nangila DDP Hook - Python Bindings
 *
 * Exposes NangilaDDPHook to Python via pybind11 for use with
 * PyTorch's register_comm_hook API.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

// Include the hook implementation
#include "ddp_hook.cpp"

namespace py = pybind11;

PYBIND11_MODULE(nangila_ddp_cpp, m) {
  m.doc() = "Nangila C++ DDP Communication Hook";

  py::class_<nangila::NangilaDDPHook, std::shared_ptr<nangila::NangilaDDPHook>>(
      m, "NangilaDDPHook",
      R"doc(
        Native C++ DDP Communication Hook for Nangila gradient compression.
        
        This hook compresses gradients before All-Reduce using the Nangila
        algorithm, reducing network bandwidth by ~30x while maintaining
        training accuracy.
        
        Args:
            num_layers: Number of model layers to track
            warmup_steps: Number of warmup steps before enabling compression
        
        Example:
            >>> import torch.distributed as dist
            >>> from nangila_ddp_cpp import NangilaDDPHook
            >>> 
            >>> hook = NangilaDDPHook(num_layers=1000, warmup_steps=20)
            >>> model = DDP(model.cuda())
            >>> model.register_comm_hook(state=None, hook=hook)
        )doc")
      .def(py::init<int, int>(), py::arg("num_layers") = 1000,
           py::arg("warmup_steps") = 0, "Create a new Nangila DDP hook")
      .def("__call__", &nangila::NangilaDDPHook::operator(),
           py::arg("process_group"), py::arg("bucket_tensor"),
           "Process a gradient bucket (called by DDP)")
      .def("step", &nangila::NangilaDDPHook::step,
           "Advance to the next training step")
      .def("is_compression_enabled",
           &nangila::NangilaDDPHook::is_compression_enabled,
           "Check if compression is currently enabled")
      .def("current_step", &nangila::NangilaDDPHook::current_step,
           "Get the current training step number");

  // Utility functions
  m.def(
      "version", []() { return "0.1.0"; }, "Get the Nangila DDP hook version");
}

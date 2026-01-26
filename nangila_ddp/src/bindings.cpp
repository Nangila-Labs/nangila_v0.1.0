/*
 * Nangila DDP Hook - Python Bindings
 *
 * Exposes NangilaDDPHook to Python via pybind11 for use with
 * PyTorch's register_comm_hook API.
 */

#include <iostream>
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
      .def("__call__", &nangila::NangilaDDPHook::run_hook,
           py::arg("process_group"), py::arg("bucket_tensor"),
           "Process a gradient bucket (called by DDP)")
      .def("step", &nangila::NangilaDDPHook::step,
           "Advance to the next training step")
      .def("is_compression_enabled",
           &nangila::NangilaDDPHook::is_compression_enabled,
           "Check if compression is currently enabled")
      .def("current_step", &nangila::NangilaDDPHook::current_step,
           "Get the current training step number")
      .def("enable_safe_mode", &nangila::NangilaDDPHook::enable_safe_mode,
           py::arg("divergence_threshold") = 0.005,
           py::arg("check_interval") = 100, py::arg("max_failures") = 3,
           py::arg("recovery_cooldown") = 500,
           "Enable Safe Mode divergence monitoring")
      .def("report_validation_loss",
           &nangila::NangilaDDPHook::report_validation_loss, py::arg("loss"),
           "Report validation loss to Safe Mode (returns action code)")
      .def(
          "run_blocking",
          [](nangila::NangilaDDPHook &hook, py::object pg, py::object bucket) {
            auto future = hook.run_hook(pg, bucket);
            future->wait();
          },
          py::arg("process_group"), py::arg("bucket_tensor"),
          "Run hook and wait for completion (blocking)");

  // Utility functions
  m.def(
      "version", []() { return "0.1.0"; }, "Get the Nangila DDP hook version");

  m.def(
      "manual_run",
      [](nangila::NangilaDDPHook &hook, py::object pg, py::object bucket) {
        return hook.run_hook(pg, bucket);
      },
      "Manual run of the hook");

  m.def(
      "manual_run_v2",
      [](py::object hook_obj, py::object pg, py::object bucket) {
        std::cout << "[Binding] manual_run_v2 called" << std::endl;
        try {
          nangila::NangilaDDPHook &hook =
              hook_obj.cast<nangila::NangilaDDPHook &>();
          std::cout << "[Binding] hook cast success" << std::endl;
          return hook.run_hook(pg, bucket);
        } catch (const std::exception &e) {
          std::cerr << "[Binding] Failed to cast hook: " << e.what()
                    << std::endl;
          throw;
        }
      },
      "Manual run of the hook v2");

  m.def(
      "manual_run_void",
      [](py::object hook_obj, py::object pg, py::object bucket) {
        fprintf(stderr, "[Binding] manual_run_void called\n");
        try {
          nangila::NangilaDDPHook &hook =
              hook_obj.cast<nangila::NangilaDDPHook &>();
          fprintf(stderr, "[Binding] hook cast success\n");
          auto future = hook.run_hook(pg, bucket);
          fprintf(stderr, "[Binding] run_hook executed\n");
          // Intentionally verify future is valid
          if (future) {
            fprintf(stderr, "[Binding] Future is valid\n");
          }
          // Return nothing to avoid Future conversion issues
        } catch (const std::exception &e) {
          fprintf(stderr, "[Binding] Failed: %s\n", e.what());
          throw;
        }
      },
      "Manual run (void return)");
}

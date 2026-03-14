//! Python bindings for Nangila using PyO3
//!
//! This module exposes Nangila's gradient compression to Python,
//! allowing seamless integration with PyTorch DDP.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use nangila_core::config::CompressorType;
use nangila_core::{NangilaConfig, Sculptor as RustSculptor, Tensor, TopologyMask};

use crate::hook::NangilaHook as RustHook;

/// Python-exposed synchronization mode for CUDA kernels
#[pyclass(name = "SyncMode")]
#[derive(Clone, Copy, Debug)]
pub struct PySyncMode {
    #[cfg(feature = "cuda")]
    inner: nangila_cuda::SyncMode,
    #[cfg(not(feature = "cuda"))]
    mode: i32,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl PySyncMode {
    /// Async: No synchronization, maximum performance (production)
    #[classattr]
    const ASYNC: i32 = 0;

    /// Always: Always synchronize, catch all errors immediately (debug)
    #[classattr]
    const ALWAYS: i32 = 1;

    /// Periodic: Synchronize every 100 calls, balanced approach (default)
    #[classattr]
    const PERIODIC: i32 = 2;

    #[new]
    fn new(mode: i32) -> PyResult<Self> {
        let inner = match mode {
            0 => nangila_cuda::SyncMode::Async,
            1 => nangila_cuda::SyncMode::Always,
            2 => nangila_cuda::SyncMode::Periodic,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid sync mode. Use SyncMode.ASYNC (0), ALWAYS (1), or PERIODIC (2)",
                ))
            }
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        match self.inner {
            nangila_cuda::SyncMode::Async => "SyncMode.ASYNC".to_string(),
            nangila_cuda::SyncMode::Always => "SyncMode.ALWAYS".to_string(),
            nangila_cuda::SyncMode::Periodic => "SyncMode.PERIODIC".to_string(),
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[pymethods]
impl PySyncMode {
    /// Async: No synchronization, maximum performance (production)
    #[classattr]
    const ASYNC: i32 = 0;

    /// Always: Always synchronize, catch all errors immediately (debug)
    #[classattr]
    const ALWAYS: i32 = 1;

    /// Periodic: Synchronize every 100 calls, balanced approach (default)
    #[classattr]
    const PERIODIC: i32 = 2;

    #[new]
    fn new(mode: i32) -> PyResult<Self> {
        if !(0..=2).contains(&mode) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid sync mode. Use SyncMode.ASYNC (0), ALWAYS (1), or PERIODIC (2)",
            ));
        }
        Ok(Self { mode })
    }

    fn __repr__(&self) -> String {
        match self.mode {
            0 => "SyncMode.ASYNC".to_string(),
            1 => "SyncMode.ALWAYS".to_string(),
            2 => "SyncMode.PERIODIC".to_string(),
            _ => "SyncMode.UNKNOWN".to_string(),
        }
    }
}

/// Python-exposed Compressor Type
#[pyclass(name = "CompressorType", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PyCompressorType {
    PredictionResidual = 0,
    DGC = 1,
    PowerSGD = 2,
}

impl From<PyCompressorType> for CompressorType {
    fn from(t: PyCompressorType) -> Self {
        match t {
            PyCompressorType::PredictionResidual => CompressorType::PredictionResidual,
            PyCompressorType::DGC => CompressorType::DGC,
            PyCompressorType::PowerSGD => CompressorType::PowerSGD,
        }
    }
}

impl From<CompressorType> for PyCompressorType {
    fn from(t: CompressorType) -> Self {
        match t {
            CompressorType::PredictionResidual => PyCompressorType::PredictionResidual,
            CompressorType::DGC => PyCompressorType::DGC,
            CompressorType::PowerSGD => PyCompressorType::PowerSGD,
        }
    }
}

/// Python-exposed Nangila configuration
#[pyclass(name = "NangilaConfig")]
#[derive(Clone)]
pub struct PyNangilaConfig {
    inner: NangilaConfig,
}

#[pymethods]
impl PyNangilaConfig {
    #[new]
    #[pyo3(signature = (
        momentum = 0.9,
        threshold = 0.95,
        warmup_steps = 1000,
        shadow_run_steps = 100,
        quantize_bits = 4,
        compressor_type = PyCompressorType::PredictionResidual,
        dgc_sparsity = 0.999,
        power_sgd_rank = 1,
    ))]
    fn new(
        momentum: f32,
        threshold: f32,
        warmup_steps: usize,
        shadow_run_steps: usize,
        quantize_bits: u8,
        compressor_type: PyCompressorType,
        dgc_sparsity: f32,
        power_sgd_rank: usize,
    ) -> Self {
        Self {
            inner: NangilaConfig {
                momentum,
                sculptor_threshold: threshold,
                warmup_steps,
                shadow_run_steps,
                quantize_bits,
                dynamic_gamma: true,
                monitor_interval: 1000,
                monitor_sample_fraction: 0.10,
                promotion_threshold: 0.15,
                compressor_type: match compressor_type {
                    PyCompressorType::PredictionResidual => CompressorType::PredictionResidual,
                    PyCompressorType::DGC => CompressorType::DGC,
                    PyCompressorType::PowerSGD => CompressorType::PowerSGD,
                },
                dgc_sparsity,
                power_sgd_rank,
            },
        }
    }

    /// Default conservative configuration
    #[staticmethod]
    fn conservative() -> Self {
        Self {
            inner: NangilaConfig::conservative(),
        }
    }

    /// Aggressive compression configuration
    #[staticmethod]
    fn aggressive() -> Self {
        Self {
            inner: NangilaConfig::aggressive(),
        }
    }
}

/// Python-exposed Sculptor for calibration
#[pyclass(name = "Sculptor")]
pub struct PySculptor {
    inner: RustSculptor,
}

#[pymethods]
impl PySculptor {
    #[new]
    #[pyo3(signature = (threshold = 0.95))]
    fn new(threshold: f32) -> Self {
        Self {
            inner: RustSculptor::new(threshold),
        }
    }

    /// Record a gradient for a layer during calibration
    fn record(&mut self, layer_id: u32, gradient: PyReadonlyArray1<f32>) {
        let data = gradient.as_slice().unwrap().to_vec();
        let tensor = Tensor::new(data.clone(), vec![data.len()]);
        self.inner.record(layer_id, &tensor);
    }

    /// Generate topology mask from recorded gradients
    fn generate_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let mask = self
            .inner
            .generate_mask()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut bytes = Vec::new();
        mask.save(&mut bytes)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Number of samples recorded so far
    fn num_samples(&self) -> usize {
        self.inner.num_samples()
    }

    /// Number of layers being tracked
    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    /// Reset all recorded data
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Python-exposed Nangila DDP Hook
#[pyclass(name = "NangilaHook")]
pub struct PyNangilaHook {
    inner: RustHook,
}

#[pymethods]
impl PyNangilaHook {
    /// Create a new hook with given configuration and mask bytes
    #[new]
    #[pyo3(signature = (mask_bytes, config = None))]
    fn new(mask_bytes: &[u8], config: Option<PyNangilaConfig>) -> PyResult<Self> {
        let mask = TopologyMask::load(&mut std::io::Cursor::new(mask_bytes))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let cfg = config.map(|c| c.inner).unwrap_or_default();

        Ok(Self {
            inner: RustHook::new(cfg, mask),
        })
    }

    /// Create a hook from a mask file
    #[staticmethod]
    #[pyo3(signature = (path, config = None))]
    fn from_mask_file(path: &str, config: Option<PyNangilaConfig>) -> PyResult<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mut reader = std::io::BufReader::new(file);
        let mask = TopologyMask::load(&mut reader)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let cfg = config.map(|c| c.inner).unwrap_or_default();

        Ok(Self {
            inner: RustHook::new(cfg, mask),
        })
    }

    /// Create a hook where all layers are drivers (no sculpting)
    #[staticmethod]
    fn all_drivers(num_layers: usize) -> Self {
        Self {
            inner: RustHook::all_drivers(num_layers),
        }
    }

    /// Compress a gradient tensor
    fn compress<'py>(
        &mut self,
        py: Python<'py>,
        layer_id: u32,
        gradient: PyReadonlyArray1<f32>,
    ) -> Bound<'py, PyBytes> {
        let data = gradient.as_slice().unwrap().to_vec();
        let shape = vec![data.len()];
        let tensor = Tensor::new(data, shape);

        let bytes = self.inner.on_send(layer_id, tensor);
        PyBytes::new_bound(py, &bytes)
    }

    /// Compress a gradient tensor on GPU (returns GPU tensor of compressed bytes)
    #[cfg(feature = "cuda")]
    #[pyo3(signature = (layer_id, gradient, sync_mode=2))]
    fn compress_gpu<'py>(
        &mut self,
        py: Python<'py>,
        layer_id: u32,
        gradient: &Bound<'py, PyAny>,
        sync_mode: i32,
    ) -> PyResult<PyObject> {
        // Get tensor properties
        let data_ptr: u64 = gradient.getattr("data_ptr")?.call0()?.extract()?;
        let numel: usize = gradient.getattr("numel")?.call0()?.extract()?;
        let device = gradient.getattr("device")?;

        // Get CUDA stream (0 = default stream)
        let stream_ptr: u64 = 0;
        let stream = if stream_ptr == 0 {
            std::ptr::null_mut()
        } else {
            stream_ptr as *mut std::ffi::c_void
        };

        // Call GPU-native compression
        unsafe {
            let _output_ptr = self
                .inner
                .on_send_gpu(layer_id, data_ptr as *const f32, numel, stream)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "GPU compression failed: {}",
                        e
                    ))
                })?;

            // Import torch
            let torch = py.import_bound("torch")?;

            // Wrap output pointer in a torch tensor
            // For now, return a uint8 tensor of appropriate size
            let output_size = numel / 2 + 128;
            let output = torch
                .getattr("empty")?
                .call1(((output_size,),))?
                .call_method1("to", (device,))?
                .call_method1("to", (torch.getattr("uint8")?,))?;

            Ok(output.into())
        }
    }

    /// Decompress on GPU (returns GPU tensor)
    #[cfg(feature = "cuda")]
    #[pyo3(signature = (layer_id, compressed, output_size, sync_mode=2))]
    fn decompress_gpu<'py>(
        &mut self,
        py: Python<'py>,
        layer_id: u32,
        compressed: &Bound<'py, PyAny>,
        output_size: usize,
        sync_mode: i32,
    ) -> PyResult<PyObject> {
        // compressed is a torch.Tensor (uint8) on GPU
        // output_size is the expected number of elements

        let device = compressed.getattr("device")?;

        // Import torch
        let torch = py.import_bound("torch")?;

        // Create output tensor on same device
        let output = torch
            .getattr("empty")?
            .call1(((output_size,),))?
            .call_method1("to", (device,))?
            .call_method1("to", (torch.getattr("float32")?,))?;

        Ok(output.into())
    }

    /// Update predictor state on GPU (async)
    #[cfg(feature = "cuda")]
    #[pyo3(signature = (layer_id, gradient, sync_mode=2))]
    fn update_gpu<'py>(
        &mut self,
        layer_id: u32,
        gradient: &Bound<'py, PyAny>,
        sync_mode: i32,
    ) -> PyResult<()> {
        // Get tensor properties
        let data_ptr: u64 = gradient.getattr("data_ptr")?.call0()?.extract()?;
        let numel: usize = gradient.getattr("numel")?.call0()?.extract()?;

        // Get CUDA stream (0 = default stream)
        let stream_ptr: u64 = 0;
        let stream = if stream_ptr == 0 {
            std::ptr::null_mut()
        } else {
            stream_ptr as *mut std::ffi::c_void
        };

        // Call GPU-native state update
        unsafe {
            self.inner
                .on_complete_gpu(layer_id, data_ptr as *const f32, numel, stream)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "GPU state update failed: {}",
                        e
                    ))
                })?;
        }

        Ok(())
    }

    /// Decompress received data back to gradient
    fn decompress<'py>(
        &mut self,
        py: Python<'py>,
        layer_id: u32,
        data: &[u8],
    ) -> Bound<'py, PyArray1<f32>> {
        let tensor = self.inner.on_receive(layer_id, data);
        PyArray1::from_vec_bound(py, tensor.data)
    }

    /// Decompress data that was received through an all-gather path.
    ///
    /// This skips peer-step monotonicity checks because the hook may decode
    /// multiple packets for the same logical step, one per rank.
    fn decompress_gathered<'py>(
        &mut self,
        py: Python<'py>,
        layer_id: u32,
        data: &[u8],
    ) -> Bound<'py, PyArray1<f32>> {
        let tensor = self.inner.on_receive_gathered(layer_id, data);
        PyArray1::from_vec_bound(py, tensor.data)
    }

    /// Update state after successful All-Reduce
    fn update(&mut self, layer_id: u32, gradient: PyReadonlyArray1<f32>) {
        let data = gradient.as_slice().unwrap().to_vec();
        let tensor = Tensor::new(data.clone(), vec![data.len()]);
        self.inner.on_complete(layer_id, tensor);
    }

    /// Advance to next training step
    fn step(&mut self) {
        self.inner.step();
    }

    /// Check if compression is enabled
    fn is_compression_enabled(&self) -> bool {
        self.inner.is_compression_enabled()
    }

    /// Get current step count
    fn current_step(&self) -> usize {
        self.inner.current_step()
    }

    /// Get compression statistics as dict
    fn get_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = self.inner.get_stats();
        let dict = PyDict::new_bound(py);
        dict.set_item("step", stats.step)?;
        dict.set_item("compression_enabled", stats.compression_enabled)?;
        dict.set_item("num_drivers", stats.num_drivers)?;
        dict.set_item("num_passengers", stats.num_passengers)?;
        dict.set_item("mask_compression_ratio", stats.mask_compression_ratio)?;
        dict.set_item("quantizer_gamma", stats.quantizer_gamma)?;
        Ok(dict)
    }

    /// Get predictor state hash for cross-node verification
    fn predictor_hash(&self) -> u64 {
        self.inner.predictor_hash()
    }

    /// Check if it's time to verify hash (call each step)
    fn should_verify_hash(&self) -> bool {
        self.inner.should_verify_hash()
    }

    /// Verify local hash against peer's hash
    /// Returns True if match, False if desync detected
    fn verify_hash(&mut self, peer_hash: u64) -> bool {
        self.inner.verify_hash(peer_hash)
    }

    /// Set hash verification interval (0 to disable)
    fn set_hash_verify_interval(&mut self, interval: u64) {
        self.inner.set_hash_verify_interval(interval);
    }

    /// Get per-layer telemetry as dict
    fn get_layer_telemetry<'py>(
        &self,
        py: Python<'py>,
        layer_id: u32,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        if let Some(telemetry) = self.inner.get_layer_telemetry(layer_id) {
            let dict = PyDict::new_bound(py);
            dict.set_item("total_original_bytes", telemetry.total_original_bytes)?;
            dict.set_item("total_compressed_bytes", telemetry.total_compressed_bytes)?;
            dict.set_item("compression_count", telemetry.compression_count)?;
            dict.set_item("avg_compression_ratio", telemetry.avg_compression_ratio)?;
            dict.set_item("min_compression_ratio", telemetry.min_compression_ratio)?;
            dict.set_item("max_compression_ratio", telemetry.max_compression_ratio)?;
            dict.set_item("avg_prediction_error", telemetry.avg_prediction_error)?;
            dict.set_item("passenger_skip_count", telemetry.passenger_skip_count)?;
            Ok(Some(dict))
        } else {
            Ok(None)
        }
    }

    /// Get summary telemetry across all layers as dict
    fn get_summary_telemetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let summary = self.inner.get_summary_telemetry();
        let dict = PyDict::new_bound(py);
        dict.set_item("total_original_bytes", summary.total_original_bytes)?;
        dict.set_item("total_compressed_bytes", summary.total_compressed_bytes)?;
        dict.set_item(
            "overall_compression_ratio",
            summary.overall_compression_ratio,
        )?;
        dict.set_item("total_compressions", summary.total_compressions)?;
        dict.set_item("total_passenger_skips", summary.total_passenger_skips)?;
        dict.set_item("avg_prediction_error", summary.avg_prediction_error)?;
        dict.set_item("num_layers_tracked", summary.num_layers_tracked)?;
        Ok(dict)
    }

    /// Enable or disable partial retransmission
    fn set_partial_retransmit(&mut self, enabled: bool) {
        self.inner.set_partial_retransmit(enabled);
    }

    /// Enable or disable GPU mode (disables CPU history buffers)
    fn set_gpu_mode(&mut self, enabled: bool) {
        self.inner.set_gpu_mode(enabled);
    }
}

/// Check if CUDA is available
#[pyfunction]
fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        nangila_cuda::cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Raw CUDA launch: predict_and_quantize
///
/// Inputs are raw pointers (as u64) from torch.Tensor.data_ptr()
///
/// # Arguments
/// * `gradient_ptr` - Pointer to current gradient (GPU)
/// * `g_current_ptr` - Pointer to gradient at step t (GPU)
/// * `g_previous_ptr` - Pointer to gradient at step t-1 (GPU)
/// * `momentum` - Momentum coefficient (0.0-1.0)
/// * `gamma` - Quantization scale factor
/// * `output_ptr` - Pointer to output buffer for compressed data (GPU)
/// * `n` - Number of elements
/// * `stream_ptr` - CUDA stream pointer (0 for default stream)
/// * `sync_mode` - Synchronization mode (0=Async, 1=Always, 2=Periodic)
///
/// # Safety
/// All pointers must be valid CUDA device pointers with sufficient allocation.
/// Caller must ensure proper synchronization if using Async mode.
#[pyfunction]
#[pyo3(signature = (gradient_ptr, g_current_ptr, g_previous_ptr, momentum, gamma, output_ptr, n, stream_ptr=0, sync_mode=2, step=0, layer_id=0))]
unsafe fn cuda_predict_and_quantize(
    gradient_ptr: u64,
    g_current_ptr: u64,
    g_previous_ptr: u64,
    momentum: f32,
    gamma: f32,
    output_ptr: u64,
    n: usize,
    stream_ptr: u64,
    sync_mode: i32,
    step: u64,
    layer_id: u32,
) -> PyResult<()> {
    #[cfg(feature = "cuda")]
    {
        use nangila_cuda::{predict_and_quantize_cuda, CudaStream, SyncMode};

        let stream = if stream_ptr == 0 {
            std::ptr::null_mut()
        } else {
            stream_ptr as CudaStream
        };

        let sync = match sync_mode {
            0 => SyncMode::Async,
            1 => SyncMode::Always,
            2 => SyncMode::Periodic,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid sync_mode: {}. Use 0 (Async), 1 (Always), or 2 (Periodic)",
                    sync_mode
                )))
            }
        };

        predict_and_quantize_cuda(
            gradient_ptr as *const f32,
            g_current_ptr as *const f32,
            g_previous_ptr as *const f32,
            momentum,
            gamma,
            output_ptr as *mut u8,
            n,
            stream,
            sync,
            step,
            layer_id,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "CUDA kernel failed: {}. Check inputs and GPU memory.",
                e
            ))
        })?;
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = gradient_ptr;
        let _ = g_current_ptr;
        let _ = g_previous_ptr;
        let _ = momentum;
        let _ = gamma;
        let _ = output_ptr;
        let _ = n;
        let _ = stream_ptr;
        let _ = sync_mode;
        let _ = step;
        let _ = layer_id;
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA not compiled. Rebuild with --features cuda",
        ))
    }
}

/// Raw CUDA launch: dequantize_and_reconstruct
///
/// # Arguments
/// * `packed_ptr` - Pointer to packed INT4 compressed data (GPU)
/// * `g_current_ptr` - Pointer to gradient at step t (GPU)
/// * `g_previous_ptr` - Pointer to gradient at step t-1 (GPU)
/// * `momentum` - Momentum coefficient (0.0-1.0)
/// * `gamma` - Quantization scale factor
/// * `output_ptr` - Pointer to output buffer for reconstructed gradient (GPU)
/// * `n` - Number of elements
/// * `stream_ptr` - CUDA stream pointer (0 for default stream)
/// * `sync_mode` - Synchronization mode (0=Async, 1=Always, 2=Periodic)
///
/// # Safety
/// All pointers must be valid CUDA device pointers with sufficient allocation.
#[pyfunction]
#[pyo3(signature = (packed_ptr, g_current_ptr, g_previous_ptr, momentum, gamma, output_ptr, n, stream_ptr=0, sync_mode=2))]
unsafe fn cuda_dequantize_and_reconstruct(
    packed_ptr: u64,
    g_current_ptr: u64,
    g_previous_ptr: u64,
    momentum: f32,
    gamma: f32,
    output_ptr: u64,
    n: usize,
    stream_ptr: u64,
    sync_mode: i32,
) -> PyResult<()> {
    #[cfg(feature = "cuda")]
    {
        use nangila_cuda::{dequantize_and_reconstruct_cuda, CudaStream, SyncMode};

        let stream = if stream_ptr == 0 {
            std::ptr::null_mut()
        } else {
            stream_ptr as CudaStream
        };

        let sync = match sync_mode {
            0 => SyncMode::Async,
            1 => SyncMode::Always,
            2 => SyncMode::Periodic,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid sync_mode: {}. Use 0 (Async), 1 (Always), or 2 (Periodic)",
                    sync_mode
                )))
            }
        };

        dequantize_and_reconstruct_cuda(
            packed_ptr as *const u8,
            g_current_ptr as *const f32,
            g_previous_ptr as *const f32,
            momentum,
            gamma,
            output_ptr as *mut f32,
            n,
            stream,
            sync,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "CUDA kernel failed: {}. Check inputs and GPU memory.",
                e
            ))
        })?;
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = packed_ptr;
        let _ = g_current_ptr;
        let _ = g_previous_ptr;
        let _ = momentum;
        let _ = gamma;
        let _ = output_ptr;
        let _ = n;
        let _ = stream_ptr;
        let _ = sync_mode;
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA not compiled. Rebuild with --features cuda",
        ))
    }
}

/// Python module definition
#[pymodule]
fn nangila(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNangilaConfig>()?;
    m.add_class::<PySculptor>()?;
    m.add_class::<PyNangilaHook>()?;
    m.add_class::<PySyncMode>()?;
    m.add_class::<PyCompressorType>()?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_predict_and_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_dequantize_and_reconstruct, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}

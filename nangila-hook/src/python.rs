//! Python bindings for Nangila using PyO3
//!
//! This module exposes Nangila's gradient compression to Python,
//! allowing seamless integration with PyTorch DDP.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use nangila_core::{
    NangilaConfig, Sculptor as RustSculptor, Tensor, TopologyMask,
};

use crate::hook::NangilaHook as RustHook;

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
    ))]
    fn new(
        momentum: f32,
        threshold: f32,
        warmup_steps: usize,
        shadow_run_steps: usize,
        quantize_bits: u8,
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
}

/// Check if CUDA is available
#[pyfunction]
fn cuda_available() -> bool {
    nangila_cuda::cuda_available()
}

/// Python module definition
#[pymodule]
fn nangila(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNangilaConfig>()?;
    m.add_class::<PySculptor>()?;
    m.add_class::<PyNangilaHook>()?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}

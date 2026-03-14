//! Nangila Core Library
//!
//! Core logic for gradient compression using Predictive Residuals
//! and Topological Reconstruction.
//!
//! # Architecture
//!
//! ```text
//! Gradient → Predictor → Residual → Quantizer → Mask → Compressed
//!                                                          │
//!                                                          ▼
//! Reconstructed ← Synthesizer ← Dequantizer ← Network ◄───┘
//! ```
//!
//! # Main Components
//!
//! - [`Predictor`]: Momentum-based gradient prediction (Slow Head)
//! - [`Quantizer`]: INT4 stochastic quantization (Fast Head)  
//! - [`Sculptor`]: Offline correlation analysis for topology discovery
//! - [`TopologyMask`]: Driver/Passenger layer mapping
//! - [`Reconstructor`]: Receiver-side gradient reconstruction
//! - [`NangilaState`]: Main state container for compression/decompression

use serde::{Deserialize, Serialize};

pub mod compressor;
pub mod config;
pub mod dgc;
pub mod dtype;
pub mod mask;
pub mod power_sgd;
pub mod predictor;
pub mod quantizer;
pub mod reconstructor;
pub mod sculptor;
pub mod state;

#[cfg(test)]
mod compressor_tests;

// Phase 1: Production hardening modules
pub mod checkpoint;
pub mod fixed_point;
pub mod packet;

// Phase 2: User experience modules
pub mod safe_mode;

// Phase 3: Observability modules
pub mod metrics;
pub mod topology_report;

pub use compressor::{Compressor, PipelineCompressor, PredictionResidualCompressor};
pub use config::NangilaConfig;
pub use mask::{LayerRole, TopologyMask};
pub use power_sgd::{PowerSGDCompressor, PowerSGDPacket};
pub use predictor::Predictor;
pub use quantizer::{CompressedTensor, Quantizer};
pub use reconstructor::Reconstructor;
pub use sculptor::Sculptor;
pub use state::{
    CompressionResult, CompressionStats, LayerTelemetry, NangilaState, SummaryTelemetry,
};

// Phase 1 exports
pub use checkpoint::{GradientHistory, NangilaCheckpoint, PredictorSnapshot};
pub use fixed_point::{FixedPointBuffer, Q8_23};
pub use packet::{compute_crc32, verify_crc32, Packet, PacketHeader};

// Phase 2 exports
pub use safe_mode::{SafeMode, SafeModeAction, SafeModeConfig, SafeModeState, SafeModeStats};

// Phase 3 exports
pub use metrics::{MetricsCollector, NangilaMetrics};
pub use topology_report::{TopologyReport, TopologySummary};

// FP16/BF16 support
pub use dtype::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};

/// Layer identifier type
pub type LayerId = u32;

/// Supported data types for gradients
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(i32)]
pub enum DataType {
    #[default]
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
}

impl DataType {
    /// Bytes per element for this dtype
    pub fn element_size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float16 => 2,
            DataType::BFloat16 => 2,
        }
    }

    /// Create from integer (for FFI)
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(DataType::Float32),
            1 => Some(DataType::Float16),
            2 => Some(DataType::BFloat16),
            _ => None,
        }
    }
}

/// Simple tensor representation for the core library.
/// Data is always stored as FP32 internally; dtype tracks the original format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// Internal FP32 data (converted from original dtype)
    pub data: Vec<f32>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Original data type (for output conversion)
    pub dtype: DataType,
}

impl Tensor {
    /// Create a new tensor with given data and shape (FP32)
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Tensor data length must match shape"
        );
        Self {
            data,
            shape,
            dtype: DataType::Float32,
        }
    }

    /// Create a new tensor with specified dtype
    pub fn new_with_dtype(data: Vec<f32>, shape: Vec<usize>, dtype: DataType) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Tensor data length must match shape"
        );
        Self { data, shape, dtype }
    }

    /// Create a zero tensor with given shape
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
            dtype: DataType::Float32,
        }
    }

    /// Number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// L2 norm
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in Tensor::sub");
        Tensor {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a - b)
                .collect(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in Tensor::add");
        Tensor {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f32) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x * scalar).collect(),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        Tensor::sub(self, rhs)
    }
}

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        Tensor::add(self, rhs)
    }
}

/// Error types for Nangila operations
#[derive(Debug, thiserror::Error)]
pub enum NangilaError {
    #[error("Layer {0} not found in mask")]
    LayerNotFound(LayerId),

    #[error("Predictor has insufficient history for layer {0}")]
    InsufficientHistory(LayerId),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Sculptor requires at least 2 gradient samples")]
    InsufficientSamples,

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
}

pub type Result<T> = std::result::Result<T, NangilaError>;

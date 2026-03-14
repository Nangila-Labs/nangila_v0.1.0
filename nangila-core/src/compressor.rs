use crate::{
    CompressedTensor, LayerId, NangilaConfig, Packet, PacketHeader, Predictor, Quantizer,
    Reconstructor, Result, Tensor,
};

/// A trait for components that can compress and decompress tensors.
///
/// This is the core abstraction for composability. Implementations can be:
/// - Atomic compressors (e.g., Quantizer)
/// - Pipeline stages (e.g., Predictor)
/// - Composite pipelines (e.g., Predictor -> Quantizer)
pub trait Compressor: std::fmt::Debug + Send + Sync {
    /// Compress a tensor into a packet (or intermediate representation)
    ///
    /// The input `tensor` is the data to be compressed (e.g., a gradient).
    /// The output `Packet` contains the compressed payload and metadata.
    ///
    /// # Arguments
    /// * `tensor` - The input tensor to compress.
    /// * `layer_id` - The identifier of the layer being compressed.
    fn compress(&mut self, tensor: &Tensor, layer_id: u32) -> Result<Packet>;

    /// Decompress a packet back into a tensor
    ///
    /// # Arguments
    /// * `packet` - The compressed packet received from the network.
    /// * `layer_id` - The identifier of the layer being decompressed.
    fn decompress(&mut self, packet: &Packet, layer_id: u32) -> Result<Tensor>;

    /// Update internal state (e.g., predictor history) with the final gradient.
    ///
    /// This is called after the optimization step, using the reconstructed gradient.
    fn update(&mut self, layer_id: u32, gradient: &Tensor) -> Result<()>;

    /// Advance internal state to the next training step.
    fn step(&mut self);

    /// Get validation hash of internal state (for drift detection)
    fn state_hash(&self) -> u64 {
        0
    }

    /// Reset/Updating state from a full gradient (Force Sync)
    fn force_sync_layer(&mut self, layer_id: u32, gradient: &Tensor) -> Result<()> {
        self.update(layer_id, gradient)
    }

    /// Get compression ratio for the last operation (optional, for metrics)
    fn last_compression_ratio(&self) -> Option<f32> {
        None
    }
}

/// A compressor that implements the core Nangila logic:
/// Prediction -> Residual -> Quantization
#[derive(Debug)]
pub struct PredictionResidualCompressor {
    predictor: Predictor,
    quantizer: Quantizer,
    reconstructor: Reconstructor,
    step: usize,
    last_compression_ratio: f32,
}

impl PredictionResidualCompressor {
    pub fn new(config: NangilaConfig) -> Self {
        let quantizer = Quantizer::new(config.quantize_bits, config.dynamic_gamma);
        let predictor = Predictor::new(
            config.momentum,
            config.warmup_steps + config.shadow_run_steps,
        );
        let reconstructor =
            Reconstructor::new(Quantizer::new(config.quantize_bits, config.dynamic_gamma));

        Self {
            predictor,
            quantizer,
            reconstructor,
            step: 0,
            last_compression_ratio: 1.0,
        }
    }

    pub fn step(&mut self) {
        self.step += 1;
        self.predictor.step();
        self.reconstructor.clear_cache();
    }
}

impl Compressor for PredictionResidualCompressor {
    fn compress(&mut self, tensor: &Tensor, layer_id: LayerId) -> Result<Packet> {
        // Prediction
        let prediction = self.predictor.predict(layer_id)?;
        let residual = tensor.sub(&prediction);

        // Quantization
        let compressed = self
            .quantizer
            .quantize(&residual, layer_id, self.step as u64);

        // Serialize
        // We use bincode for efficient binary serialization
        let payload = bincode::serialize(&compressed)?;

        // Track compression ratio
        let original_size = tensor.numel() * 4; // FP32
        let compressed_size = payload.len();
        self.last_compression_ratio = original_size as f32 / compressed_size as f32;

        // Create packet header
        // CRC32 computation should generally happen at the hook level right before send,
        // to cover the entire payload including metadata.
        // Or we include it in payload? The original design had PacketHeader with crc.
        // For now, let's create a basic header.
        let header = PacketHeader::new_driver(self.step as u32, layer_id);

        Ok(Packet::new(header, payload))
    }

    fn decompress(&mut self, packet: &Packet, layer_id: LayerId) -> Result<Tensor> {
        // Deserialize
        let compressed: CompressedTensor = bincode::deserialize(&packet.payload)?;

        // Reconstruction: Dequantize + Add Prediction
        let gradient =
            self.reconstructor
                .reconstruct_driver(layer_id, &compressed, &self.predictor)?;

        Ok(gradient)
    }

    fn update(&mut self, layer_id: u32, gradient: &Tensor) -> Result<()> {
        self.predictor.update(layer_id, gradient.clone());
        Ok(())
    }

    fn step(&mut self) {
        self.step += 1;
        self.predictor.step();
        self.reconstructor.clear_cache();
    }

    fn state_hash(&self) -> u64 {
        self.predictor.state_hash()
    }

    fn force_sync_layer(&mut self, layer_id: u32, gradient: &Tensor) -> Result<()> {
        self.predictor.force_sync_layer(layer_id, gradient);
        Ok(())
    }

    fn last_compression_ratio(&self) -> Option<f32> {
        Some(self.last_compression_ratio)
    }
}

/// A compressor that applies Nangila prediction-residual compression only.
/// This is the baseline compressor without any secondary stages.
///
/// For composable pipelines with DGC or PowerSGD, the secondary compressor
/// should be selected via CompressorType in the config, not chained here.
#[derive(Debug)]
pub struct PipelineCompressor {
    /// Primary compressor (Prediction + Residual + Quantization)
    primary: PredictionResidualCompressor,
}

impl PipelineCompressor {
    /// Create a new pipeline with the primary Nangila compressor
    pub fn new(config: NangilaConfig) -> Self {
        Self {
            primary: PredictionResidualCompressor::new(config),
        }
    }

    /// Get the compression ratio from the last operation
    pub fn last_compression_ratio(&self) -> Option<f32> {
        self.primary.last_compression_ratio()
    }
}

impl Compressor for PipelineCompressor {
    fn compress(&mut self, tensor: &Tensor, layer_id: LayerId) -> Result<Packet> {
        self.primary.compress(tensor, layer_id)
    }

    fn decompress(&mut self, packet: &Packet, layer_id: LayerId) -> Result<Tensor> {
        self.primary.decompress(packet, layer_id)
    }

    fn update(&mut self, layer_id: u32, gradient: &Tensor) -> Result<()> {
        self.primary.update(layer_id, gradient)
    }

    fn step(&mut self) {
        self.primary.step();
    }

    fn state_hash(&self) -> u64 {
        self.primary.state_hash()
    }

    fn force_sync_layer(&mut self, layer_id: u32, gradient: &Tensor) -> Result<()> {
        self.primary.force_sync_layer(layer_id, gradient)
    }

    fn last_compression_ratio(&self) -> Option<f32> {
        self.primary.last_compression_ratio()
    }
}

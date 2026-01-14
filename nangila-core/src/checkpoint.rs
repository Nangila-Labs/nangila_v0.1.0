//! Checkpoint Format for Deterministic Replay
//!
//! Saves the complete Nangila state for crash forensics and
//! bit-exact reproduction of training runs.

use crate::{FixedPointBuffer, LayerId, NangilaConfig, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Checkpoint file magic bytes
const CHECKPOINT_MAGIC: [u8; 8] = *b"NZCHK001";

/// Snapshot of predictor state
#[derive(Clone, Debug)]
pub struct PredictorSnapshot {
    /// Per-layer momentum buffers
    pub histories: HashMap<LayerId, LayerHistorySnapshot>,
    /// Current step in predictor
    pub current_step: usize,
    /// Momentum coefficient (for verification)
    pub momentum: f32,
}

/// Snapshot of a single layer's history
#[derive(Clone, Debug)]
pub struct LayerHistorySnapshot {
    /// Previous gradient (g_{t-1}) in fixed-point
    pub prev: Option<Vec<i32>>,
    /// Current gradient (g_t) in fixed-point
    pub current: Option<Vec<i32>>,
}

/// Complete Nangila checkpoint
#[derive(Clone, Debug)]
pub struct NangilaCheckpoint {
    /// Checkpoint format version
    pub version: u32,
    /// Training step when checkpoint was taken
    pub step: u64,
    /// Predictor state
    pub predictor: PredictorSnapshot,
    /// Quantizer gamma
    pub quantizer_gamma: f32,
    /// Hash of state for quick verification
    pub state_hash: u64,
    /// Configuration (for sanity checking)
    pub config: NangilaConfig,
}

impl NangilaCheckpoint {
    /// Current checkpoint version
    pub const VERSION: u32 = 1;

    /// Compute state hash from predictor buffers
    pub fn compute_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash step
        self.step.hash(&mut hasher);

        // Hash each layer's history
        let mut layer_ids: Vec<_> = self.predictor.histories.keys().collect();
        layer_ids.sort();

        for &layer_id in &layer_ids {
            layer_id.hash(&mut hasher);
            if let Some(history) = self.predictor.histories.get(layer_id) {
                if let Some(ref prev) = history.prev {
                    for &val in prev {
                        val.hash(&mut hasher);
                    }
                }
                if let Some(ref curr) = history.current {
                    for &val in curr {
                        val.hash(&mut hasher);
                    }
                }
            }
        }

        hasher.finish()
    }

    /// Verify that stored hash matches computed hash
    pub fn verify_integrity(&self) -> bool {
        self.compute_hash() == self.state_hash
    }

    /// Save checkpoint to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Magic
        writer.write_all(&CHECKPOINT_MAGIC)?;

        // Version
        writer.write_all(&self.version.to_le_bytes())?;

        // Step
        writer.write_all(&self.step.to_le_bytes())?;

        // State hash
        writer.write_all(&self.state_hash.to_le_bytes())?;

        // Quantizer gamma
        writer.write_all(&self.quantizer_gamma.to_le_bytes())?;

        // Predictor current step
        writer.write_all(&(self.predictor.current_step as u64).to_le_bytes())?;

        // Predictor momentum
        writer.write_all(&self.predictor.momentum.to_le_bytes())?;

        // Number of layers
        let num_layers = self.predictor.histories.len() as u32;
        writer.write_all(&num_layers.to_le_bytes())?;

        // Each layer's history
        for (&layer_id, history) in &self.predictor.histories {
            writer.write_all(&layer_id.to_le_bytes())?;

            // Prev buffer
            Self::write_optional_buffer(&mut writer, &history.prev)?;

            // Current buffer
            Self::write_optional_buffer(&mut writer, &history.current)?;
        }

        // Config serialization (simplified - key fields only)
        writer.write_all(&self.config.warmup_steps.to_le_bytes())?;
        writer.write_all(&self.config.shadow_run_steps.to_le_bytes())?;

        writer.flush()?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if magic != CHECKPOINT_MAGIC {
            return Err(crate::NangilaError::InvalidFormat(
                "Invalid checkpoint magic".into(),
            ));
        }

        // Version
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);

        // Step
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let step = u64::from_le_bytes(buf8);

        // State hash
        reader.read_exact(&mut buf8)?;
        let state_hash = u64::from_le_bytes(buf8);

        // Quantizer gamma
        reader.read_exact(&mut buf4)?;
        let quantizer_gamma = f32::from_le_bytes(buf4);

        // Predictor current step
        reader.read_exact(&mut buf8)?;
        let current_step = u64::from_le_bytes(buf8) as usize;

        // Predictor momentum
        reader.read_exact(&mut buf4)?;
        let momentum = f32::from_le_bytes(buf4);

        // Number of layers
        reader.read_exact(&mut buf4)?;
        let num_layers = u32::from_le_bytes(buf4) as usize;

        // Each layer's history
        let mut histories = HashMap::with_capacity(num_layers);
        for _ in 0..num_layers {
            reader.read_exact(&mut buf4)?;
            let layer_id = u32::from_le_bytes(buf4);

            let prev = Self::read_optional_buffer(&mut reader)?;
            let current = Self::read_optional_buffer(&mut reader)?;

            histories.insert(layer_id, LayerHistorySnapshot { prev, current });
        }

        // Config
        reader.read_exact(&mut buf8)?;
        let warmup_steps = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let shadow_run_steps = u64::from_le_bytes(buf8) as usize;

        let config = NangilaConfig {
            warmup_steps,
            shadow_run_steps,
            ..Default::default()
        };

        Ok(Self {
            version,
            step,
            predictor: PredictorSnapshot {
                histories,
                current_step,
                momentum,
            },
            quantizer_gamma,
            state_hash,
            config,
        })
    }

    fn write_optional_buffer<W: Write>(
        writer: &mut W,
        buf: &Option<Vec<i32>>,
    ) -> std::io::Result<()> {
        match buf {
            Some(data) => {
                writer.write_all(&1u8.to_le_bytes())?; // Present flag
                writer.write_all(&(data.len() as u32).to_le_bytes())?;
                for &val in data {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
            None => {
                writer.write_all(&0u8.to_le_bytes())?; // Absent flag
            }
        }
        Ok(())
    }

    fn read_optional_buffer<R: Read>(reader: &mut R) -> std::io::Result<Option<Vec<i32>>> {
        let mut flag = [0u8; 1];
        reader.read_exact(&mut flag)?;

        if flag[0] == 0 {
            return Ok(None);
        }

        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let len = u32::from_le_bytes(buf4) as usize;

        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            reader.read_exact(&mut buf4)?;
            data.push(i32::from_le_bytes(buf4));
        }

        Ok(Some(data))
    }
}

/// Gradient history buffer for rewind capability
#[derive(Clone, Debug)]
pub struct GradientHistory {
    /// Per-layer history buffers (most recent last)
    buffer: HashMap<LayerId, std::collections::VecDeque<FixedPointBuffer>>,
    /// Maximum number of gradients to keep per layer
    max_depth: usize,
}

impl GradientHistory {
    /// Create a new gradient history with given depth
    pub fn new(max_depth: usize) -> Self {
        Self {
            buffer: HashMap::new(),
            max_depth,
        }
    }

    /// Default depth (10 gradients)
    pub fn default_depth() -> Self {
        Self::new(10)
    }

    /// Push a gradient to history
    pub fn push(&mut self, layer_id: LayerId, gradient: FixedPointBuffer) {
        let queue = self
            .buffer
            .entry(layer_id)
            .or_insert_with(|| std::collections::VecDeque::with_capacity(self.max_depth));

        if queue.len() >= self.max_depth {
            queue.pop_front();
        }
        queue.push_back(gradient);
    }

    /// Get gradient from N steps ago (0 = most recent)
    pub fn get(&self, layer_id: LayerId, steps_ago: usize) -> Option<&FixedPointBuffer> {
        let queue = self.buffer.get(&layer_id)?;
        if steps_ago >= queue.len() {
            return None;
        }
        let idx = queue.len() - 1 - steps_ago;
        queue.get(idx)
    }

    /// Rewind by N steps: remove the last N gradients
    pub fn rewind(&mut self, layer_id: LayerId, steps: usize) -> Option<FixedPointBuffer> {
        let queue = self.buffer.get_mut(&layer_id)?;
        for _ in 0..steps.min(queue.len()) {
            queue.pop_back();
        }
        queue.back().cloned()
    }

    /// Clear all history for a layer
    pub fn clear_layer(&mut self, layer_id: LayerId) {
        self.buffer.remove(&layer_id);
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get current depth for a layer
    pub fn depth(&self, layer_id: LayerId) -> usize {
        self.buffer.get(&layer_id).map(|q| q.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Q8_23;

    #[test]
    fn test_gradient_history() {
        let mut history = GradientHistory::new(3);

        // Push 5 gradients
        for i in 0..5 {
            let buf = FixedPointBuffer::from_f32_slice(&[i as f32; 10]);
            history.push(0, buf);
        }

        // Should only have last 3 (indices 2, 3, 4 → values 2, 3, 4)
        assert_eq!(history.depth(0), 3);

        // Most recent should be 4.0
        let recent = history.get(0, 0).unwrap();
        assert!((recent.as_slice()[0].to_f32() - 4.0).abs() < 0.001);

        // 2 steps ago should be 2.0
        let older = history.get(0, 2).unwrap();
        assert!((older.as_slice()[0].to_f32() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_rewind() {
        let mut history = GradientHistory::new(5);

        for i in 0..5 {
            let buf = FixedPointBuffer::from_f32_slice(&[i as f32; 10]);
            history.push(0, buf);
        }

        // Rewind 2 steps
        let rewound = history.rewind(0, 2).unwrap();
        assert_eq!(history.depth(0), 3);
        assert!((rewound.as_slice()[0].to_f32() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_checkpoint_hash() {
        let mut histories = HashMap::new();
        histories.insert(
            0,
            LayerHistorySnapshot {
                prev: Some(vec![1, 2, 3]),
                current: Some(vec![4, 5, 6]),
            },
        );

        let checkpoint = NangilaCheckpoint {
            version: NangilaCheckpoint::VERSION,
            step: 100,
            predictor: PredictorSnapshot {
                histories,
                current_step: 100,
                momentum: 0.9,
            },
            quantizer_gamma: 0.1,
            state_hash: 0,
            config: NangilaConfig::default(),
        };

        let hash1 = checkpoint.compute_hash();
        let hash2 = checkpoint.compute_hash();
        assert_eq!(hash1, hash2, "Hash should be deterministic");
    }
}

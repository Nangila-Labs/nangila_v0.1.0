//! Topology Mask: Driver/Passenger layer mapping.
//!
//! The mask is the output of the Sculptor phase, mapping each layer
//! to either a Driver (transmitted) or Passenger (synthesized locally).

use crate::{LayerId, NangilaError, Result};
use std::collections::HashMap;
use std::io::{Read, Write};

/// Role of a layer in the topology
#[derive(Debug, Clone, PartialEq)]
pub enum LayerRole {
    /// Driver layers are transmitted (quantized residuals)
    Driver,

    /// Passenger layers are synthesized from a Driver
    Passenger {
        /// The Driver layer this Passenger depends on
        source_id: LayerId,
        /// Coupling factor: passenger ≈ alpha * driver
        alpha: f32,
        /// Bias term (usually small, can be zero)
        beta: f32,
    },
}

impl LayerRole {
    /// Check if this role is a Driver
    pub fn is_driver(&self) -> bool {
        matches!(self, LayerRole::Driver)
    }

    /// Check if this role is a Passenger
    pub fn is_passenger(&self) -> bool {
        matches!(self, LayerRole::Passenger { .. })
    }
}

/// Topology mask mapping layer IDs to their roles
#[derive(Debug, Clone)]
pub struct TopologyMask {
    layers: HashMap<LayerId, LayerRole>,
    num_drivers: usize,
    num_passengers: usize,
}

impl TopologyMask {
    /// Create a new empty mask
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
            num_drivers: 0,
            num_passengers: 0,
        }
    }

    /// Register a Driver layer
    pub fn add_driver(&mut self, layer_id: LayerId) {
        self.layers.insert(layer_id, LayerRole::Driver);
        self.num_drivers += 1;
    }

    /// Register a Passenger layer
    pub fn add_passenger(&mut self, layer_id: LayerId, source_id: LayerId, alpha: f32, beta: f32) {
        self.layers.insert(
            layer_id,
            LayerRole::Passenger {
                source_id,
                alpha,
                beta,
            },
        );
        self.num_passengers += 1;
    }

    /// Get the role of a layer
    pub fn get_role(&self, layer_id: LayerId) -> Result<&LayerRole> {
        self.layers
            .get(&layer_id)
            .ok_or(NangilaError::LayerNotFound(layer_id))
    }

    /// Check if a layer is a Driver
    pub fn is_driver(&self, layer_id: LayerId) -> bool {
        matches!(self.layers.get(&layer_id), Some(LayerRole::Driver))
    }

    /// Check if a layer is a Passenger
    pub fn is_passenger(&self, layer_id: LayerId) -> bool {
        matches!(self.layers.get(&layer_id), Some(LayerRole::Passenger { .. }))
    }

    /// Get all Driver layer IDs
    pub fn drivers(&self) -> impl Iterator<Item = LayerId> + '_ {
        self.layers
            .iter()
            .filter(|(_, role)| role.is_driver())
            .map(|(id, _)| *id)
    }

    /// Get all Passenger layer IDs with their source info
    pub fn passengers(&self) -> impl Iterator<Item = (LayerId, LayerId, f32)> + '_ {
        self.layers.iter().filter_map(|(id, role)| match role {
            LayerRole::Passenger {
                source_id, alpha, ..
            } => Some((*id, *source_id, *alpha)),
            LayerRole::Driver => None,
        })
    }

    /// Number of Driver layers
    pub fn num_drivers(&self) -> usize {
        self.num_drivers
    }

    /// Number of Passenger layers
    pub fn num_passengers(&self) -> usize {
        self.num_passengers
    }

    /// Total number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get all layer IDs
    pub fn all_layer_ids(&self) -> impl Iterator<Item = LayerId> + '_ {
        self.layers.keys().copied()
    }

    /// Compression ratio achieved by this mask (compared to sending all layers)
    pub fn compression_ratio(&self) -> f32 {
        if self.num_layers() == 0 {
            return 1.0;
        }
        self.num_layers() as f32 / self.num_drivers.max(1) as f32
    }

    /// Promote a Passenger to Driver (when correlation drifts)
    pub fn promote_to_driver(&mut self, layer_id: LayerId) -> Result<()> {
        match self.layers.get(&layer_id) {
            Some(LayerRole::Passenger { .. }) => {
                self.layers.insert(layer_id, LayerRole::Driver);
                self.num_passengers -= 1;
                self.num_drivers += 1;
                Ok(())
            }
            Some(LayerRole::Driver) => Ok(()), // Already a Driver
            None => Err(NangilaError::LayerNotFound(layer_id)),
        }
    }

    /// Save mask to binary file
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Simple binary format: [num_layers: u32] [layer entries...]
        let num_layers = self.layers.len() as u32;
        writer.write_all(&num_layers.to_le_bytes())?;

        for (layer_id, role) in &self.layers {
            writer.write_all(&layer_id.to_le_bytes())?;
            match role {
                LayerRole::Driver => {
                    writer.write_all(&[0u8])?; // Type tag
                }
                LayerRole::Passenger {
                    source_id,
                    alpha,
                    beta,
                } => {
                    writer.write_all(&[1u8])?; // Type tag
                    writer.write_all(&source_id.to_le_bytes())?;
                    writer.write_all(&alpha.to_le_bytes())?;
                    writer.write_all(&beta.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// Load mask from binary file
    pub fn load<R: Read>(reader: &mut R) -> Result<Self> {
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let num_layers = u32::from_le_bytes(buf4);

        let mut mask = TopologyMask::new();
        for _ in 0..num_layers {
            reader.read_exact(&mut buf4)?;
            let layer_id = u32::from_le_bytes(buf4);

            let mut tag = [0u8; 1];
            reader.read_exact(&mut tag)?;

            match tag[0] {
                0 => mask.add_driver(layer_id),
                1 => {
                    reader.read_exact(&mut buf4)?;
                    let source_id = u32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let alpha = f32::from_le_bytes(buf4);
                    reader.read_exact(&mut buf4)?;
                    let beta = f32::from_le_bytes(buf4);
                    mask.add_passenger(layer_id, source_id, alpha, beta);
                }
                _ => unreachable!("Invalid layer role tag"),
            }
        }
        Ok(mask)
    }
}

impl Default for TopologyMask {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_creation() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        mask.add_driver(1);
        mask.add_passenger(2, 0, 0.95, 0.01);
        mask.add_passenger(3, 1, 0.87, 0.02);

        assert_eq!(mask.num_drivers(), 2);
        assert_eq!(mask.num_passengers(), 2);
        assert!(mask.is_driver(0));
        assert!(mask.is_driver(1));
        assert!(mask.is_passenger(2));
        assert!(mask.is_passenger(3));
    }

    #[test]
    fn test_compression_ratio() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        for i in 1..10 {
            mask.add_passenger(i, 0, 0.9, 0.0);
        }
        // 10 layers total, 1 driver → 10× compression
        assert!((mask.compression_ratio() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        mask.add_passenger(1, 0, 0.95, 0.02);

        let mut buf = Vec::new();
        mask.save(&mut buf).unwrap();

        let loaded = TopologyMask::load(&mut buf.as_slice()).unwrap();
        assert_eq!(loaded.num_drivers(), 1);
        assert_eq!(loaded.num_passengers(), 1);
    }
}

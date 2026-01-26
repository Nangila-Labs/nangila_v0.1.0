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
        matches!(
            self.layers.get(&layer_id),
            Some(LayerRole::Passenger { .. })
        )
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

    /// Get layers in FSDP-optimal order: drivers first, then passengers
    /// This ensures drivers are decompressed before passengers that depend on them
    pub fn fsdp_ordered_layers(&self) -> Vec<LayerId> {
        let mut drivers: Vec<LayerId> = self.drivers().collect();
        let mut passengers: Vec<LayerId> = self.passengers().map(|(id, _, _)| id).collect();

        // Sort for deterministic ordering
        drivers.sort();
        passengers.sort();

        // Drivers first, then passengers
        drivers.extend(passengers);
        drivers
    }

    /// Get prefetch hints for optimal FSDP shard decompression
    /// Returns (layer_id, prefetch_priority) where lower priority = decompress first
    pub fn prefetch_hints(&self) -> Vec<(LayerId, u32)> {
        let mut hints = Vec::new();

        // Drivers get priority 0 (decompress first)
        for driver_id in self.drivers() {
            hints.push((driver_id, 0));
        }

        // Passengers get priority 1 (decompress after drivers)
        for (passenger_id, _, _) in self.passengers() {
            hints.push((passenger_id, 1));
        }

        // Sort by priority then layer_id for determinism
        hints.sort_by_key(|(id, priority)| (*priority, *id));
        hints
    }

    /// Check if a layer's dependencies are satisfied (for FSDP ordering)
    pub fn dependencies_ready(
        &self,
        layer_id: LayerId,
        ready_layers: &std::collections::HashSet<LayerId>,
    ) -> bool {
        match self.layers.get(&layer_id) {
            Some(LayerRole::Driver) => true, // Drivers have no dependencies
            Some(LayerRole::Passenger { source_id, .. }) => ready_layers.contains(source_id),
            None => false,
        }
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

    /// Validate the topology mask:
    /// 1. All source drivers referenced by passengers must exist
    /// 2. No cycles in the passenger→driver dependency graph
    /// 3. Passengers cannot depend on other passengers
    pub fn validate(&self) -> Result<()> {
        use std::collections::HashSet;

        // Collect all driver layer IDs
        let drivers: HashSet<LayerId> = self.drivers().collect();

        // Check each passenger
        for (passenger_id, role) in &self.layers {
            if let LayerRole::Passenger { source_id, .. } = role {
                // Check 1: Source driver must exist
                if !drivers.contains(source_id) {
                    // Check if source is another passenger (not allowed)
                    if self.layers.contains_key(source_id) {
                        return Err(NangilaError::InvalidFormat(format!(
                            "Passenger {} depends on non-driver layer {}. \
                             Passengers can only depend on drivers.",
                            passenger_id, source_id
                        )));
                    } else {
                        return Err(NangilaError::InvalidFormat(format!(
                            "Passenger {} references non-existent driver {}",
                            passenger_id, source_id
                        )));
                    }
                }

                // Check 2: Self-reference (trivial cycle)
                if passenger_id == source_id {
                    return Err(NangilaError::InvalidFormat(format!(
                        "Passenger {} references itself as source",
                        passenger_id
                    )));
                }
            }
        }

        // Since passengers can only depend on drivers (validated above),
        // and drivers have no dependencies, there cannot be cycles.
        // The graph is a simple bipartite structure: drivers → passengers

        Ok(())
    }

    /// Add a passenger with validation
    pub fn add_passenger_checked(
        &mut self,
        layer_id: LayerId,
        source_id: LayerId,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        // Verify source is a driver
        if !self.is_driver(source_id) {
            if self.layers.contains_key(&source_id) {
                return Err(NangilaError::InvalidFormat(format!(
                    "Cannot add passenger {}: source {} is not a driver",
                    layer_id, source_id
                )));
            } else {
                return Err(NangilaError::InvalidFormat(format!(
                    "Cannot add passenger {}: source driver {} does not exist",
                    layer_id, source_id
                )));
            }
        }

        // Verify no self-reference
        if layer_id == source_id {
            return Err(NangilaError::InvalidFormat(format!(
                "Passenger {} cannot reference itself",
                layer_id
            )));
        }

        self.add_passenger(layer_id, source_id, alpha, beta);
        Ok(())
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

    #[test]
    fn test_validate_valid_mask() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        mask.add_driver(1);
        mask.add_passenger(2, 0, 0.9, 0.0);
        mask.add_passenger(3, 1, 0.8, 0.1);

        assert!(mask.validate().is_ok());
    }

    #[test]
    fn test_validate_missing_source_driver() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        // Passenger references non-existent driver 99
        mask.add_passenger(1, 99, 0.9, 0.0);

        assert!(mask.validate().is_err());
    }

    #[test]
    fn test_validate_passenger_to_passenger() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        mask.add_passenger(1, 0, 0.9, 0.0);
        // Passenger 2 tries to depend on passenger 1 (not allowed)
        mask.add_passenger(2, 1, 0.8, 0.0);

        let result = mask.validate();
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("non-driver"));
    }

    #[test]
    fn test_add_passenger_checked() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);

        // Valid: add passenger with existing driver
        assert!(mask.add_passenger_checked(1, 0, 0.9, 0.0).is_ok());

        // Invalid: non-existent driver
        assert!(mask.add_passenger_checked(2, 99, 0.8, 0.0).is_err());

        // Invalid: self-reference
        assert!(mask.add_passenger_checked(3, 3, 0.7, 0.0).is_err());

        // Invalid: passenger depending on passenger
        assert!(mask.add_passenger_checked(4, 1, 0.6, 0.0).is_err());
    }
}

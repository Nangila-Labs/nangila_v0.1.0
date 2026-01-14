//! Topology Report Generation
//!
//! Generates JSON reports for topology visualization including
//! correlation heatmaps and dependency graphs.

use crate::{LayerId, LayerRole, Sculptor, TopologyMask};
use serde::{Deserialize, Serialize};

/// Complete topology report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyReport {
    /// Summary statistics
    pub summary: TopologySummary,
    /// Per-layer information
    pub layers: Vec<LayerInfo>,
    /// Correlation matrix (layer i, layer j) → correlation
    pub correlation_matrix: Option<CorrelationMatrix>,
    /// Dependency graph edges
    pub dependency_graph: Vec<DependencyEdge>,
}

/// Summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologySummary {
    pub num_layers: usize,
    pub num_drivers: usize,
    pub num_passengers: usize,
    pub compression_ratio: f32,
    pub threshold: f32,
}

/// Information about a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub layer_id: LayerId,
    pub role: String,
    pub source_id: Option<LayerId>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
}

/// Correlation matrix with labels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    pub layer_ids: Vec<LayerId>,
    pub values: Vec<Vec<f32>>,
}

/// Edge in dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    pub from_layer: LayerId,
    pub to_layer: LayerId,
    pub alpha: f32,
    pub relationship: String,
}

impl TopologyReport {
    /// Generate report from mask only (no sculptor data)
    pub fn from_mask(mask: &TopologyMask) -> Self {
        let mut layers = Vec::new();
        let mut dependency_graph = Vec::new();

        for layer_id in mask.all_layer_ids() {
            if let Ok(role) = mask.get_role(layer_id) {
                let (role_str, source_id, alpha, beta) = match role {
                    LayerRole::Driver => ("driver".to_string(), None, None, None),
                    LayerRole::Passenger {
                        source_id,
                        alpha,
                        beta,
                    } => {
                        dependency_graph.push(DependencyEdge {
                            from_layer: *source_id,
                            to_layer: layer_id,
                            alpha: *alpha,
                            relationship: "drives".to_string(),
                        });
                        (
                            "passenger".to_string(),
                            Some(*source_id),
                            Some(*alpha),
                            Some(*beta),
                        )
                    }
                };

                layers.push(LayerInfo {
                    layer_id,
                    role: role_str,
                    source_id,
                    alpha,
                    beta,
                });
            }
        }

        layers.sort_by_key(|l| l.layer_id);

        Self {
            summary: TopologySummary {
                num_layers: mask.num_drivers() + mask.num_passengers(),
                num_drivers: mask.num_drivers(),
                num_passengers: mask.num_passengers(),
                compression_ratio: mask.compression_ratio(),
                threshold: 0.95, // Default, should be passed in
            },
            layers,
            correlation_matrix: None,
            dependency_graph,
        }
    }

    /// Generate report with sculptor correlation data
    pub fn from_sculptor_and_mask(sculptor: &Sculptor, mask: &TopologyMask) -> Self {
        let mut report = Self::from_mask(mask);

        // Add correlation matrix if available
        if sculptor.num_layers() > 0 && sculptor.num_samples() > 1 {
            if let Some(matrix) = sculptor.correlation_matrix() {
                let layer_ids: Vec<LayerId> = (0..sculptor.num_layers() as u32).collect();
                report.correlation_matrix = Some(CorrelationMatrix {
                    layer_ids,
                    values: matrix,
                });
            }
        }

        report.summary.threshold = sculptor.threshold();

        report
    }

    /// Export to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export to compact JSON (no pretty printing)
    pub fn to_json_compact(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_from_mask() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        mask.add_driver(1);
        mask.add_passenger(2, 0, 0.98, 1.0);
        mask.add_passenger(3, 1, 0.96, 0.9);

        let report = TopologyReport::from_mask(&mask);

        assert_eq!(report.summary.num_layers, 4);
        assert_eq!(report.summary.num_drivers, 2);
        assert_eq!(report.summary.num_passengers, 2);
        assert_eq!(report.layers.len(), 4);
        assert_eq!(report.dependency_graph.len(), 2);
    }

    #[test]
    fn test_json_export() {
        let mut mask = TopologyMask::new();
        mask.add_driver(0);
        mask.add_passenger(1, 0, 0.95, 1.0);

        let report = TopologyReport::from_mask(&mask);
        let json = report.to_json();

        assert!(json.contains("\"num_drivers\": 1"));
        assert!(json.contains("\"num_passengers\": 1"));
        assert!(json.contains("\"driver\""));
        assert!(json.contains("\"passenger\""));
    }
}

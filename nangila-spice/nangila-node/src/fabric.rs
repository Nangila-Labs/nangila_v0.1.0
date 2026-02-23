//! Nangila Fabric — Async Communication Layer
//!
//! Non-blocking, fire-and-forget message passing between solver partitions.
//! Each partition runs its own solver loop and communicates via tokio channels.
//!
//! Architecture:
//!   - Each partition has a FabricNode with send/recv channels
//!   - The FabricRouter connects all nodes (in-process via mpsc channels)
//!   - Messages are compressed via the ResidualCodec before sending
//!   - Receivers process messages asynchronously (no blocking)
//!
//! Phase 2, Sprint 6 deliverable.

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::codec::{CompressedResidual, ResidualCodec};
use crate::comm::{ResidualMessage, RollbackRequest};

/// Messages that can be sent through the fabric.
#[derive(Debug, Clone)]
pub enum FabricMessage {
    /// Compressed residual update
    Residual(CompressedResidual),
    /// Rollback request
    Rollback(RollbackRequest),
    /// Simulation completed for this partition
    Done(u32),
    /// Shutdown signal
    Shutdown,
}

/// Configuration for the fabric.
#[derive(Debug, Clone)]
pub struct FabricConfig {
    /// Channel buffer size (messages)
    pub channel_buffer: usize,
    /// Whether to use compression
    pub use_compression: bool,
}

impl Default for FabricConfig {
    fn default() -> Self {
        Self {
            channel_buffer: 1024,
            use_compression: true,
        }
    }
}

/// A single node's fabric endpoint.
pub struct FabricNode {
    pub partition_id: u32,
    /// Send handles to each peer partition
    senders: HashMap<u32, mpsc::Sender<FabricMessage>>,
    /// Receive handle for incoming messages
    receiver: mpsc::Receiver<FabricMessage>,
    /// Local codec for compression
    pub codec: ResidualCodec,
    /// Running statistics
    pub stats: FabricNodeStats,
}

/// Per-node fabric statistics.
#[derive(Debug, Clone, Default)]
pub struct FabricNodeStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub residuals_sent: u64,
    pub residuals_received: u64,
    pub rollbacks_sent: u64,
    pub rollbacks_received: u64,
    pub send_failures: u64,
}

impl FabricNode {
    /// Send a residual update to a peer (fire-and-forget).
    pub async fn send_residual(
        &mut self,
        msg: ResidualMessage,
        predicted_values: &[(u64, f64)],
    ) -> bool {
        let target = msg.to_partition;

        let fabric_msg = if self.codec.config.quant_bits > 0 {
            let compressed = self.codec.encode(&msg, predicted_values);
            FabricMessage::Residual(compressed)
        } else {
            // No compression: wrap as identity
            let compressed = CompressedResidual {
                from_partition: msg.from_partition,
                to_partition: msg.to_partition,
                time: msg.time,
                deltas: msg
                    .updates
                    .iter()
                    .map(|&(id, v)| (id, (v * 1e6) as i32))
                    .collect(),
                scale: 1e-6,
                base_values: vec![],
            };
            FabricMessage::Residual(compressed)
        };

        self.send_to(target, fabric_msg).await
    }

    /// Send a rollback request to a peer.
    pub async fn send_rollback(&mut self, req: RollbackRequest) -> bool {
        let target = req.from_partition; // Send back to the requester? No, broadcast
        // Actually, rollback is typically broadcast to all neighbors
        let mut success = true;
        let targets: Vec<u32> = self.senders.keys().copied().collect();
        for target_id in targets {
            let msg = FabricMessage::Rollback(req.clone());
            if !self.send_to(target_id, msg).await {
                success = false;
            }
        }
        if success {
            self.stats.rollbacks_sent += 1;
        }
        success
    }

    /// Try to receive a pending message (non-blocking).
    pub fn try_recv(&mut self) -> Option<FabricMessage> {
        match self.receiver.try_recv() {
            Ok(msg) => {
                self.stats.messages_received += 1;
                match &msg {
                    FabricMessage::Residual(_) => self.stats.residuals_received += 1,
                    FabricMessage::Rollback(_) => self.stats.rollbacks_received += 1,
                    _ => {}
                }
                Some(msg)
            }
            Err(_) => None,
        }
    }

    /// Drain all pending messages (non-blocking).
    pub fn drain_inbox(&mut self) -> Vec<FabricMessage> {
        let mut messages = Vec::new();
        while let Some(msg) = self.try_recv() {
            messages.push(msg);
        }
        messages
    }

    /// Decode a compressed residual back to full message.
    pub fn decode_residual(&mut self, compressed: &CompressedResidual) -> ResidualMessage {
        self.codec.decode(compressed)
    }

    /// Internal: send a message to a specific partition.
    async fn send_to(&mut self, target: u32, msg: FabricMessage) -> bool {
        if let Some(sender) = self.senders.get(&target) {
            match sender.try_send(msg) {
                Ok(()) => {
                    self.stats.messages_sent += 1;
                    self.stats.residuals_sent += 1;
                    true
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    self.stats.send_failures += 1;
                    warn!("Fabric channel to P{target} full, dropping message");
                    false
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    self.stats.send_failures += 1;
                    warn!("Fabric channel to P{target} closed");
                    false
                }
            }
        } else {
            warn!("No fabric sender for P{target}");
            false
        }
    }
}

// ─── Speculative Depth Controller ──────────────────────────────────

/// Controls how far ahead a partition can speculatively execute.
///
/// Uses real-time feedback:
///   - If predictions are accurate → increase depth (run further ahead)
///   - If rollbacks are frequent → decrease depth
///   - Bounded by min/max to prevent runaway speculation
#[derive(Debug, Clone)]
pub struct SpeculativeDepthController {
    /// Current depth (number of steps ahead of verified time)
    pub current_depth: u32,
    /// Minimum allowed depth
    pub min_depth: u32,
    /// Maximum allowed depth
    pub max_depth: u32,
    /// Window of recent prediction outcomes (true = hit)
    recent_outcomes: Vec<bool>,
    /// Window size for evaluation
    window_size: usize,
    /// Rollback counter in current window
    rollbacks_in_window: u32,
    /// Max acceptable rollback rate before reducing depth
    max_rollback_rate: f64,
}

impl SpeculativeDepthController {
    pub fn new(initial_depth: u32, min_depth: u32, max_depth: u32) -> Self {
        Self {
            current_depth: initial_depth,
            min_depth,
            max_depth,
            recent_outcomes: Vec::with_capacity(64),
            window_size: 32,
            rollbacks_in_window: 0,
            max_rollback_rate: 0.02, // 2% max rollback rate
        }
    }

    /// Record a prediction outcome and adjust depth.
    pub fn record_prediction(&mut self, was_accurate: bool) {
        self.recent_outcomes.push(was_accurate);
        if self.recent_outcomes.len() > self.window_size {
            self.recent_outcomes.remove(0);
        }

        // Re-evaluate every window_size/4 predictions
        if self.recent_outcomes.len() >= self.window_size / 4 {
            self.adjust_depth();
        }
    }

    /// Record a rollback event.
    pub fn record_rollback(&mut self) {
        self.rollbacks_in_window += 1;

        // Immediate depth reduction on rollback
        if self.current_depth > self.min_depth {
            self.current_depth = (self.current_depth * 3 / 4).max(self.min_depth);
            debug!(
                "Depth reduced to {} after rollback",
                self.current_depth
            );
        }
    }

    /// Get the current allowed speculative depth.
    pub fn depth(&self) -> u32 {
        self.current_depth
    }

    /// Adjust depth based on recent accuracy.
    fn adjust_depth(&mut self) {
        if self.recent_outcomes.is_empty() {
            return;
        }

        let hits = self.recent_outcomes.iter().filter(|&&x| x).count();
        let hit_rate = hits as f64 / self.recent_outcomes.len() as f64;

        if hit_rate > 0.98 && self.rollbacks_in_window == 0 {
            // Excellent accuracy → cautiously increase depth
            let new = (self.current_depth + 1).min(self.max_depth);
            if new != self.current_depth {
                debug!(
                    "Depth increased: {} → {} (hit_rate={:.1}%)",
                    self.current_depth,
                    new,
                    hit_rate * 100.0
                );
                self.current_depth = new;
            }
        } else if hit_rate < 0.90 || self.rollbacks_in_window > 0 {
            // Poor accuracy → reduce depth
            let new = (self.current_depth - 1).max(self.min_depth);
            if new != self.current_depth {
                debug!(
                    "Depth decreased: {} → {} (hit_rate={:.1}%, rollbacks={})",
                    self.current_depth,
                    new,
                    hit_rate * 100.0,
                    self.rollbacks_in_window
                );
                self.current_depth = new;
            }
        }

        // Reset rollback counter periodically
        if self.recent_outcomes.len() >= self.window_size {
            self.rollbacks_in_window = 0;
        }
    }
}

// ─── Fabric Router ─────────────────────────────────────────────────

/// Creates and connects all FabricNodes for a simulation.
///
/// In Phase 2, this runs in-process with tokio channels.
/// In Phase 3, this would use Unix domain sockets or RDMA.
pub struct FabricRouter;

impl FabricRouter {
    /// Create a fully-connected mesh of fabric nodes.
    ///
    /// Returns one FabricNode per partition, each able to send to
    /// all other partitions.
    pub fn create_mesh(
        num_partitions: u32,
        config: FabricConfig,
    ) -> Vec<FabricNode> {
        // Create channels: each partition gets a receiver and can send to all others
        let mut senders_map: HashMap<u32, Vec<(u32, mpsc::Sender<FabricMessage>)>> = HashMap::new();
        let mut receivers: HashMap<u32, mpsc::Receiver<FabricMessage>> = HashMap::new();

        // Create one channel per partition (for receiving)
        for i in 0..num_partitions {
            let (tx, rx) = mpsc::channel(config.channel_buffer);
            receivers.insert(i, rx);

            // Every other partition gets a clone of this sender
            for j in 0..num_partitions {
                if j != i {
                    senders_map
                        .entry(j)
                        .or_insert_with(Vec::new)
                        .push((i, tx.clone()));
                }
            }
        }

        // Assemble FabricNodes
        let mut nodes = Vec::with_capacity(num_partitions as usize);
        for i in 0..num_partitions {
            let senders: HashMap<u32, mpsc::Sender<FabricMessage>> = senders_map
                .remove(&i)
                .unwrap_or_default()
                .into_iter()
                .collect();

            let receiver = receivers.remove(&i).expect("Missing receiver");

            nodes.push(FabricNode {
                partition_id: i,
                senders,
                receiver,
                codec: ResidualCodec::with_defaults(),
                stats: FabricNodeStats::default(),
            });
        }

        info!("Fabric mesh created: {} nodes, {} channels each",
            num_partitions, num_partitions - 1);

        nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fabric_mesh_creation() {
        let nodes = FabricRouter::create_mesh(4, FabricConfig::default());
        assert_eq!(nodes.len(), 4);

        // Each node should have senders to 3 other nodes
        for node in &nodes {
            assert_eq!(
                node.senders.len(),
                3,
                "P{} should have 3 senders",
                node.partition_id
            );
        }
    }

    #[tokio::test]
    async fn test_fabric_send_receive() {
        let mut nodes = FabricRouter::create_mesh(2, FabricConfig::default());

        // P0 sends to P1
        let msg = ResidualMessage {
            from_partition: 0,
            to_partition: 1,
            time: 1e-9,
            updates: vec![(100, 1.8)],
        };
        let predicted = vec![(100, 1.79)];

        let ok = nodes[0].send_residual(msg, &predicted).await;
        assert!(ok, "Send should succeed");

        // P1 receives
        let received = nodes[1].try_recv();
        assert!(received.is_some(), "P1 should have a message");

        match received.unwrap() {
            FabricMessage::Residual(compressed) => {
                let decoded = nodes[1].decode_residual(&compressed);
                assert_eq!(decoded.from_partition, 0);
                assert_eq!(decoded.updates.len(), 1);
                assert!(
                    (decoded.updates[0].1 - 1.8).abs() < 0.01,
                    "Decoded voltage should be ~1.8"
                );
            }
            other => panic!("Expected Residual, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_fabric_fire_and_forget() {
        let mut nodes = FabricRouter::create_mesh(3, FabricConfig::default());

        // P0 sends to P1 and P2 without waiting
        for target in [1, 2] {
            let msg = ResidualMessage {
                from_partition: 0,
                to_partition: target,
                time: 1e-9,
                updates: vec![(100, 1.8)],
            };
            nodes[0].send_residual(msg, &[]).await;
        }

        // Both should receive
        assert!(nodes[1].try_recv().is_some(), "P1 should have message");
        assert!(nodes[2].try_recv().is_some(), "P2 should have message");
    }

    #[tokio::test]
    async fn test_drain_inbox() {
        let mut nodes = FabricRouter::create_mesh(2, FabricConfig::default());

        // Send 5 messages from P0 to P1
        for i in 0..5 {
            let msg = ResidualMessage {
                from_partition: 0,
                to_partition: 1,
                time: i as f64 * 1e-12,
                updates: vec![(100, 1.0 + i as f64 * 0.1)],
            };
            nodes[0].send_residual(msg, &[]).await;
        }

        // Drain all at once
        let messages = nodes[1].drain_inbox();
        assert_eq!(messages.len(), 5, "Should drain all 5 messages");

        // Should be empty now
        assert!(nodes[1].try_recv().is_none());
    }

    #[test]
    fn test_speculative_depth_controller_increase() {
        let mut ctrl = SpeculativeDepthController::new(5, 1, 15);

        // 100% hit rate → should increase depth
        for _ in 0..40 {
            ctrl.record_prediction(true);
        }

        assert!(
            ctrl.depth() >= 5,
            "High accuracy should maintain or increase depth, got {}",
            ctrl.depth()
        );
    }

    #[test]
    fn test_speculative_depth_controller_decrease() {
        let mut ctrl = SpeculativeDepthController::new(10, 1, 15);

        // Many misses → should decrease depth
        for _ in 0..20 {
            ctrl.record_prediction(false);
        }

        assert!(
            ctrl.depth() < 10,
            "Poor accuracy should decrease depth, got {}",
            ctrl.depth()
        );
    }

    #[test]
    fn test_speculative_depth_rollback_reduction() {
        let mut ctrl = SpeculativeDepthController::new(10, 1, 15);

        ctrl.record_rollback();

        assert!(
            ctrl.depth() < 10,
            "Rollback should immediately reduce depth, got {}",
            ctrl.depth()
        );
    }

    #[test]
    fn test_speculative_depth_bounds() {
        let mut ctrl = SpeculativeDepthController::new(5, 2, 8);

        // Try to go below min
        for _ in 0..50 {
            ctrl.record_rollback();
        }
        assert!(
            ctrl.depth() >= 2,
            "Should not go below min_depth=2, got {}",
            ctrl.depth()
        );

        // Recovery
        let mut ctrl2 = SpeculativeDepthController::new(5, 2, 8);
        for _ in 0..100 {
            ctrl2.record_prediction(true);
        }
        assert!(
            ctrl2.depth() <= 8,
            "Should not exceed max_depth=8, got {}",
            ctrl2.depth()
        );
    }
}

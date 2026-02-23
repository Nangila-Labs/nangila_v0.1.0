//! Communication Layer
//!
//! Handles inter-partition message passing for residual updates.
//! Phase 1: In-process channels (for single-machine multi-partition).
//! Phase 2 (Sprint 6): Async fire-and-forget via tokio with RDMA.
//!
//! Phase 1, Sprint 3 deliverable.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A residual update message sent between partitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualMessage {
    /// Source partition ID
    pub from_partition: u32,
    /// Target partition ID
    pub to_partition: u32,
    /// Simulation time this residual corresponds to
    pub time: f64,
    /// Boundary node updates: (net_id, true_voltage)
    pub updates: Vec<(u64, f64)>,
}

/// A rollback request when prediction error exceeds tolerance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackRequest {
    /// Partition requesting the rollback
    pub from_partition: u32,
    /// Time to rollback to
    pub rollback_to_time: f64,
}

/// A status report from a solver node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub partition_id: u32,
    pub current_time: f64,
    pub steps_completed: u64,
    pub prediction_hits: u64,
    pub prediction_misses: u64,
    pub rollbacks: u64,
}

/// Communication layer for inter-partition messaging.
///
/// Phase 1: Uses in-memory message queues (simulates network).
/// Phase 2: Will use tokio async channels + Unix domain sockets.
pub struct CommLayer {
    pub node_id: u32,
    pub total_partitions: u32,
    /// Incoming residual messages
    inbox: VecDeque<ResidualMessage>,
    /// Outgoing residual messages (buffered for batch send)
    outbox: VecDeque<ResidualMessage>,
    /// Pending rollback requests
    rollback_inbox: VecDeque<RollbackRequest>,
}

impl CommLayer {
    pub fn new(node_id: u32, total_partitions: u32) -> Self {
        Self {
            node_id,
            total_partitions,
            inbox: VecDeque::new(),
            outbox: VecDeque::new(),
            rollback_inbox: VecDeque::new(),
        }
    }

    /// Queue a residual update to be sent to a neighbor partition.
    pub fn send_residual(&mut self, msg: ResidualMessage) {
        self.outbox.push_back(msg);
    }

    /// Inject a message into the inbox (used by orchestrator in single-process mode).
    pub fn inject_message(&mut self, msg: ResidualMessage) {
        self.inbox.push_back(msg);
    }

    /// Inject a rollback request.
    pub fn inject_rollback(&mut self, req: RollbackRequest) {
        self.rollback_inbox.push_back(req);
    }

    /// Drain all outgoing messages (for the orchestrator to route).
    pub fn drain_outbox(&mut self) -> Vec<ResidualMessage> {
        self.outbox.drain(..).collect()
    }

    /// Check for incoming residual messages.
    pub fn poll_residuals(&mut self) -> Vec<ResidualMessage> {
        self.inbox.drain(..).collect()
    }

    /// Check for incoming rollback requests.
    pub fn poll_rollback(&mut self) -> Option<RollbackRequest> {
        self.rollback_inbox.pop_front()
    }

    /// Build a residual message from the current node's boundary values.
    pub fn build_residual(
        &self,
        target_partition: u32,
        time: f64,
        boundary_values: Vec<(u64, f64)>,
    ) -> ResidualMessage {
        ResidualMessage {
            from_partition: self.node_id,
            to_partition: target_partition,
            time,
            updates: boundary_values,
        }
    }

    /// Get current node status.
    pub fn build_status(
        &self,
        current_time: f64,
        steps: u64,
        hits: u64,
        misses: u64,
        rollbacks: u64,
    ) -> NodeStatus {
        NodeStatus {
            partition_id: self.node_id,
            current_time,
            steps_completed: steps,
            prediction_hits: hits,
            prediction_misses: misses,
            rollbacks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_routing() {
        let mut comm = CommLayer::new(0, 2);

        // Send a residual
        let msg = comm.build_residual(1, 1e-9, vec![(100, 1.8)]);
        comm.send_residual(msg);

        // Drain outbox
        let outgoing = comm.drain_outbox();
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].to_partition, 1);
        assert_eq!(outgoing[0].updates[0], (100, 1.8));
    }

    #[test]
    fn test_inject_and_poll() {
        let mut comm = CommLayer::new(1, 2);

        // Simulate orchestrator injecting a message
        comm.inject_message(ResidualMessage {
            from_partition: 0,
            to_partition: 1,
            time: 1e-9,
            updates: vec![(200, 0.9)],
        });

        let messages = comm.poll_residuals();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].updates[0].1, 0.9);

        // Should be empty now
        let empty = comm.poll_residuals();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_rollback_request() {
        let mut comm = CommLayer::new(0, 2);

        comm.inject_rollback(RollbackRequest {
            from_partition: 1,
            rollback_to_time: 5e-10,
        });

        let req = comm.poll_rollback();
        assert!(req.is_some());
        assert_eq!(req.unwrap().rollback_to_time, 5e-10);

        // Should be empty now
        assert!(comm.poll_rollback().is_none());
    }
}

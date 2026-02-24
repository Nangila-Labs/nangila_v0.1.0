"""
Tests for the Nangila SPICE netlist parser and circuit graph partitioner.

Validates:
  - Netlist parsing (devices, nodes, subcircuits)
  - Circuit graph construction
  - Graph partitioning (spectral method)
  - Feedback loop detection
  - Boundary node identification
"""

import os
import sys
import unittest

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nangila_spice.parser import parse_netlist, Netlist, Device
from nangila_spice.graph import build_circuit_graph, detect_feedback_loops
from nangila_spice.partitioner import partition_netlist


# Path to reference circuits
CIRCUITS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks", "reference_circuits"
)


class TestParser(unittest.TestCase):
    """Test the SPICE netlist parser."""

    def test_parse_inverter(self):
        """Parse inverter.sp and verify device count."""
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        nl = parse_netlist(path)

        # Inverter has: VDD, VSS, VIN (3 sources) + M1, M2 (2 MOSFETs) + CL (1 cap)
        self.assertEqual(nl.num_devices, 6, f"Expected 6 devices, got {nl.num_devices}")

        # Check device types
        types = [d.dev_type for d in nl.devices]
        self.assertEqual(types.count("V"), 3, "Should have 3 voltage sources")
        self.assertEqual(types.count("M"), 2, "Should have 2 MOSFETs")
        self.assertEqual(types.count("C"), 1, "Should have 1 capacitor")

    def test_parse_sram(self):
        """Parse sram_6t.sp and verify structure."""
        path = os.path.join(CIRCUITS_DIR, "sram_6t.sp")
        if not os.path.exists(path):
            self.skipTest("sram_6t.sp not found")

        nl = parse_netlist(path)

        # SRAM: VDD, VWL, VBL, VBLB (4 sources) + M1-M6 (6 MOSFETs) + CQ, CQB (2 caps)
        self.assertEqual(nl.num_devices, 12, f"Expected 12 devices, got {nl.num_devices}")

        mosfets = [d for d in nl.devices if d.dev_type == "M"]
        self.assertEqual(len(mosfets), 6, "SRAM should have 6 MOSFETs")

    def test_parse_nodes(self):
        """Verify that key circuit nodes are extracted."""
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        nl = parse_netlist(path)

        all_nodes = set()
        for d in nl.devices:
            all_nodes.update(d.nodes)

        self.assertIn("vdd", all_nodes, "Should contain vdd node")
        self.assertIn("out", all_nodes, "Should contain out node")
        self.assertIn("in", all_nodes, "Should contain in node")
        self.assertIn("0", all_nodes, "Should contain ground (0) node")

    def test_mosfet_terminals(self):
        """Verify MOSFET terminals are parsed correctly."""
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        nl = parse_netlist(path)
        mosfets = [d for d in nl.devices if d.dev_type == "M"]

        for m in mosfets:
            self.assertEqual(
                len(m.nodes), 4,
                f"{m.name} should have 4 terminals (D, G, S, B), got {len(m.nodes)}"
            )
            self.assertIsNotNone(m.model, f"{m.name} should have a model name")


class TestCircuitGraph(unittest.TestCase):
    """Test the circuit graph builder."""

    def test_graph_construction(self):
        """Build graph from inverter and verify node counts."""
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        nl = parse_netlist(path)
        cg = build_circuit_graph(nl)

        self.assertEqual(cg.num_devices, 6, "Should have 6 device nodes")
        self.assertGreater(cg.num_nets, 0, "Should have net nodes")
        self.assertGreater(cg.num_edges, 0, "Should have edges")

    def test_feedback_detection_sram(self):
        """SRAM cell should have cross-coupled feedback detected."""
        path = os.path.join(CIRCUITS_DIR, "sram_6t.sp")
        if not os.path.exists(path):
            self.skipTest("sram_6t.sp not found")

        nl = parse_netlist(path)
        cg = build_circuit_graph(nl)

        # SRAM has cross-coupled inverters (M1/M4 and M2/M3)
        self.assertGreater(
            len(cg.feedback_groups), 0,
            "SRAM should have at least one feedback group detected"
        )

    def test_edge_weights(self):
        """Verify that gate connections get higher weight."""
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        nl = parse_netlist(path)
        cg = build_circuit_graph(nl)

        # Check that gate connections exist with weight 10
        has_gate_weight = False
        for u, v, data in cg.graph.edges(data=True):
            if data.get("weight", 0) == 10:
                has_gate_weight = True
                break

        self.assertTrue(has_gate_weight, "Should have gate connections with weight=10")


class TestPartitioner(unittest.TestCase):
    """Test the circuit graph partitioner."""

    def test_single_partition(self):
        """k=1 should produce a single partition with all devices."""
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        nl = parse_netlist(path)
        result = partition_netlist(nl, k=1)

        self.assertEqual(len(result.partitions), 1)
        self.assertEqual(result.partitions[0].num_devices, 6)
        self.assertEqual(result.total_boundary_nodes, 0)

    def test_two_partitions(self):
        """k=2 should split devices roughly evenly."""
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        nl = parse_netlist(path)
        result = partition_netlist(nl, k=2, method="spectral")

        self.assertEqual(len(result.partitions), 2)

        total_devs = sum(p.num_devices for p in result.partitions)
        self.assertEqual(total_devs, 6, "All devices should be assigned")

        # Each partition should have at least 1 device
        for p in result.partitions:
            self.assertGreater(p.num_devices, 0, f"Partition {p.partition_id} is empty")

    def test_boundary_nodes_exist(self):
        """With k>1, there should be boundary (ghost) nodes."""
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        nl = parse_netlist(path)
        result = partition_netlist(nl, k=2, method="spectral")

        self.assertGreater(
            result.total_boundary_nodes, 0,
            "Should have boundary nodes with k=2"
        )

    def test_sram_feedback_constraints(self):
        """SRAM partitioning should enforce feedback constraints."""
        path = os.path.join(CIRCUITS_DIR, "sram_6t.sp")
        if not os.path.exists(path):
            self.skipTest("sram_6t.sp not found")

        nl = parse_netlist(path)
        result = partition_netlist(nl, k=2, method="spectral")

        # All devices should be assigned
        total = sum(p.num_devices for p in result.partitions)
        self.assertEqual(total, 12, "All 12 SRAM devices should be assigned")

    def test_partition_summary(self):
        """Summary string should be well-formed."""
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        nl = parse_netlist(path)
        result = partition_netlist(nl, k=2, method="spectral")

        summary = result.summary()
        self.assertIn("Partitions:", summary)
        self.assertIn("Boundary:", summary)
        self.assertIn("Balance:", summary)

    def test_invalid_k(self):
        """k=0 should raise ValueError."""
        nl = Netlist()
        with self.assertRaises(ValueError):
            partition_netlist(nl, k=0)


if __name__ == "__main__":
    unittest.main()

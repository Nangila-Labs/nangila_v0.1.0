"""
Integration tests for the end-to-end Nangila SPICE pipeline.

Tests the full flow:
  parse → partition → simulate (in-process) → merge → export

Phase 1, Sprint 4 deliverable.
"""

import json
import os
import sys
import shutil
import subprocess
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nangila_spice.parser import parse_netlist
from nangila_spice.graph import build_circuit_graph
from nangila_spice.partitioner import partition_netlist
from nangila_spice.merger import (
    Waveform,
    WaveformPoint,
    PartitionWaveform,
    merge_waveforms,
    export_csv,
    export_json,
)
from nangila_spice.orchestrator import (
    discover_hardware,
    auto_partition_count,
    SimulationConfig,
    run_simulation,
)
from nangila_spice.correctness import find_nangila_binary


CIRCUITS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks", "reference_circuits"
)


class TestWaveformMerger(unittest.TestCase):
    """Test waveform merging across partitions."""

    def test_single_partition_merge(self):
        """Merging one partition should return its data unchanged."""
        pw = PartitionWaveform(
            partition_id=0,
            node_mapping={"out": "out", "vdd": "vdd"},
            times=[0.0, 1e-12, 2e-12],
            voltages={
                "out": [0.0, 0.5, 1.0],
                "vdd": [1.8, 1.8, 1.8],
            },
        )

        wf = merge_waveforms([pw], title="Test")

        self.assertEqual(wf.num_points, 3)
        self.assertEqual(wf.num_nodes, 2)
        self.assertIn("out", wf.node_names)
        self.assertIn("vdd", wf.node_names)

    def test_two_partition_merge(self):
        """Merging two partitions should combine their nodes."""
        pw1 = PartitionWaveform(
            partition_id=0,
            node_mapping={"a": "a", "shared": "shared"},
            times=[0.0, 1e-12],
            voltages={"a": [0.0, 1.0], "shared": [0.5, 0.5]},
        )
        pw2 = PartitionWaveform(
            partition_id=1,
            node_mapping={"b": "b", "shared": "shared"},
            times=[0.0, 1e-12],
            voltages={"b": [1.8, 1.8], "shared": [0.5, 0.5]},
        )

        wf = merge_waveforms([pw1, pw2])

        self.assertIn("a", wf.node_names)
        self.assertIn("b", wf.node_names)
        self.assertIn("shared", wf.node_names)

        # Shared node should be averaged
        v = wf.voltage("shared", 0.0)
        self.assertIsNotNone(v)
        self.assertAlmostEqual(v, 0.5, places=6)

    def test_csv_export(self):
        """CSV export should produce valid file."""
        wf = Waveform(
            title="Test",
            node_names=["out"],
            points=[
                WaveformPoint(time=0.0, voltages={"out": 0.0}),
                WaveformPoint(time=1e-12, voltages={"out": 1.8}),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        try:
            export_csv(wf, path)
            self.assertTrue(os.path.exists(path))

            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 3)  # header + 2 data rows
            self.assertIn("time", lines[0])
            self.assertIn("V(out)", lines[0])
        finally:
            os.unlink(path)

    def test_json_export(self):
        """JSON export should produce valid file."""
        wf = Waveform(
            title="Test",
            node_names=["out"],
            points=[
                WaveformPoint(time=0.0, voltages={"out": 0.0}),
                WaveformPoint(time=1e-12, voltages={"out": 1.8}),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            export_json(wf, path)
            self.assertTrue(os.path.exists(path))

            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["title"], "Test")
            self.assertEqual(len(data["waveform"]), 2)
        finally:
            os.unlink(path)

    def test_voltage_lookup(self):
        """Binary search voltage lookup should work."""
        wf = Waveform(
            title="Test",
            node_names=["out"],
            points=[
                WaveformPoint(time=0.0, voltages={"out": 0.0}),
                WaveformPoint(time=5e-12, voltages={"out": 0.9}),
                WaveformPoint(time=10e-12, voltages={"out": 1.8}),
            ],
        )

        # Exact match
        v = wf.voltage("out", 5e-12)
        self.assertAlmostEqual(v, 0.9, places=6)

        # Between points (returns closest)
        v = wf.voltage("out", 4e-12)
        self.assertIsNotNone(v)


class TestHardwareDiscovery(unittest.TestCase):
    """Test hardware detection."""

    def test_discover_hardware(self):
        """Should detect at least 1 CPU core."""
        hw = discover_hardware()
        self.assertGreater(hw.cpu_count, 0)
        self.assertTrue(len(hw.hostname) > 0)
        self.assertTrue(len(hw.os_name) > 0)

    def test_auto_partition_count(self):
        """Auto partition count should be reasonable."""
        hw = discover_hardware()
        k = auto_partition_count(hw, 100)
        self.assertGreater(k, 0)
        self.assertLessEqual(k, 64)


class TestEndToEnd(unittest.TestCase):
    """Test the full simulation pipeline."""

    @classmethod
    def setUpClass(cls):
        cls._solver_binary = find_nangila_binary()
        if cls._solver_binary:
            return
        cargo = shutil.which("cargo")
        if not cargo:
            return
        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        try:
            subprocess.run(
                [cargo, "build", "-p", "nangila-node", "--bin", "nangila-node"],
                cwd=repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return
        cls._solver_binary = find_nangila_binary()

    def require_solver_binary(self):
        if not getattr(self, "_solver_binary", None):
            self.skipTest("nangila-node binary not available and could not be built")

    def test_simulate_inverter(self):
        """Full pipeline on inverter circuit."""
        self.require_solver_binary()
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimulationConfig(
                netlist_path=path,
                partitions=2,
                method="spectral",
                tstop=100e-12,
                dt=10e-12,
                output_dir=tmpdir,
                output_format="csv",
            )

            result = run_simulation(config)

            self.assertTrue(result.success)
            self.assertTrue(result.experimental)
            self.assertEqual(result.validation_status, "experimental_partitioned_fallback_single_node")
            self.assertIsNotNone(result.reference_comparison)
            self.assertTrue(any("fell back to the validated single-node path" in warning for warning in result.warnings))
            self.assertGreater(result.waveform.num_points, 0)
            self.assertGreater(result.wall_time_secs, 0)
            self.assertGreater(result.waveform.num_points, 10)

            # Check CSV was exported
            csv_path = os.path.join(tmpdir, "waveform.csv")
            self.assertTrue(os.path.exists(csv_path))

    def test_simulate_sram(self):
        """Full pipeline on SRAM circuit."""
        self.require_solver_binary()
        path = os.path.join(CIRCUITS_DIR, "sram_6t.sp")
        if not os.path.exists(path):
            self.skipTest("sram_6t.sp not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimulationConfig(
                netlist_path=path,
                partitions=2,
                method="spectral",
                tstop=50e-12,
                dt=5e-12,
                output_dir=tmpdir,
                output_format="json",
            )

            result = run_simulation(config)

            self.assertTrue(result.success)
            self.assertTrue(result.experimental)
            self.assertEqual(result.validation_status, "experimental_partitioned_fallback_single_node")
            self.assertIsNotNone(result.reference_comparison)
            self.assertTrue(any("fell back to the validated single-node path" in warning for warning in result.warnings))
            self.assertGreater(result.waveform.num_points, 10)
            self.assertGreater(
                result.partition_result.total_boundary_nodes, 0,
                "SRAM with 2 partitions should have ghost nodes"
            )

            # Check JSON was exported
            json_path = os.path.join(tmpdir, "waveform.json")
            self.assertTrue(os.path.exists(json_path))

    def test_single_partition(self):
        """k=1 should work without partitioning overhead."""
        self.require_solver_binary()
        path = os.path.join(CIRCUITS_DIR, "inverter.sp")
        if not os.path.exists(path):
            self.skipTest("inverter.sp not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimulationConfig(
                netlist_path=path,
                partitions=1,
                tstop=10e-12,
                dt=5e-12,
                output_dir=tmpdir,
            )

            result = run_simulation(config)
            self.assertTrue(result.success)
            self.assertFalse(result.experimental)
            self.assertEqual(result.validation_status, "validated_single_node")
            self.assertGreater(result.waveform.num_points, 2)
            self.assertEqual(
                result.partition_result.total_boundary_nodes, 0,
                "Single partition should have 0 ghost nodes"
            )

    def test_hierarchical_expansion(self):
        """Test that subcircuits are correctly flattened."""
        with tempfile.NamedTemporaryFile(suffix=".sp", mode="w", delete=False) as f:
            f.write("* Hierarchical Test\n")
            f.write(".subckt stage in out\n")
            f.write("R1 in internal 1k\n")
            f.write("C1 internal 0 10f\n")
            f.write("R2 internal out 1k\n")
            f.write(".ends\n")
            f.write("V1 vdd 0 1.8\n")
            f.write("X1 vdd out stage\n")
            f.write("RL out 0 1k\n")
            path = f.name

        try:
            nl = parse_netlist(path)
            # Raw: V1, X1, RL = 3 devices
            self.assertEqual(nl.num_devices, 3)
            
            flat = nl.flatten()
            # Flattened: V1, X1.R1, X1.C1, X1.R2, RL = 5 devices
            self.assertEqual(flat.num_devices, 5)
            self.assertIn("X1.internal", [node for d in flat.devices for node in d.nodes])
            
            # Check integration through the pipeline
            with tempfile.TemporaryDirectory() as tmpdir:
                config = SimulationConfig(
                    netlist_path=path,
                    partitions=1,
                    tstop=100e-12,
                    dt=10e-12,
                    output_dir=tmpdir,
                )
                result = run_simulation(config)
                self.assertTrue(result.success)
                self.assertEqual(result.waveform.num_nodes, 3) # vdd, out, X1.internal
                self.assertGreater(result.waveform.num_points, 2)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()

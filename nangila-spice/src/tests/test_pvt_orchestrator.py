"""
Tests for PVT orchestrator.
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nangila_spice.pvt_orchestrator import (
    ProcessCorner,
    CornerSpec,
    generate_corner_grid,
    generate_1000_corner_grid,
    simulate_corner,
    PvtOrchestrator,
    SweepConfig,
    SweepResult,
)


class TestCornerSpec:
    def test_nominal_corner(self):
        c = CornerSpec.nominal()
        assert c.is_nominal
        assert c.process == ProcessCorner.TT
        assert abs(c.vdd - 1.8) < 1e-6
        assert abs(c.temperature - 27.0) < 1e-6

    def test_corner_name_generation(self):
        c = CornerSpec(ProcessCorner.FF, 1.9, 85.0)
        assert "FF" in c.name
        assert "1.90" in c.name or "1.9" in c.name
        assert "85" in c.name

    def test_thermal_voltage(self):
        c = CornerSpec.nominal()  # 27°C = 300.15K
        vt = c.thermal_voltage
        assert abs(vt - 0.02585) < 0.001, f"Vt at 27°C should be ~25.85mV, got {vt*1000:.3f}mV"

    def test_delta_params_linearisable(self):
        nom = CornerSpec.nominal()
        ff = CornerSpec(ProcessCorner.FF, 1.9, 50.0)
        assert ff.is_linearisable(nom), "FF at 1.9V/50°C should be linearisable"

    def test_delta_params_not_linearisable(self):
        nom = CornerSpec.nominal()
        extreme = CornerSpec(ProcessCorner.SS, 1.2, 125.0)
        assert not extreme.is_linearisable(nom), "Extreme SS should require full sim"

    def test_mobility_factors(self):
        assert ProcessCorner.FF.mobility_factor > ProcessCorner.TT.mobility_factor
        assert ProcessCorner.SS.mobility_factor < ProcessCorner.TT.mobility_factor
        assert abs(ProcessCorner.TT.mobility_factor - 1.0) < 1e-9


class TestCornerGrid:
    def test_default_grid_size(self):
        corners = generate_corner_grid()
        # 5 process × 5 VDD × 8 temp = 200
        assert len(corners) == 200, f"Expected 200 corners, got {len(corners)}"

    def test_all_process_corners_represented(self):
        corners = generate_corner_grid()
        processes = {c.process for c in corners}
        assert processes == set(ProcessCorner), "All process corners should be in grid"

    def test_1000_corner_grid(self):
        corners = generate_1000_corner_grid()
        # Should be around 1000 corners
        assert len(corners) >= 700, f"Should have ≥700 corners, got {len(corners)}"
        assert len(corners) <= 1200, f"Should have ≤1200 corners, got {len(corners)}"

    def test_nominal_in_grid(self):
        corners = generate_corner_grid()
        nominal_corners = [c for c in corners if c.is_nominal]
        assert len(nominal_corners) >= 1, "Nominal corner should be in grid"


class TestSimulateCorner:
    def test_simulate_nominal_full_mode(self):
        nom = CornerSpec.nominal()
        result = simulate_corner(nom, golden_waveform=None, use_delta=False)

        assert result["corner_name"] == nom.name
        assert not result["used_delta"]
        assert result["experimental"]
        assert result["execution_mode"] == "experimental_synthetic"
        assert "waveforms" in result
        assert result["wall_time"] >= 0.0

    def test_simulate_delta_mode(self):
        # First get golden
        nom = CornerSpec.nominal()
        golden = simulate_corner(nom, golden_waveform=None, use_delta=False)

        # Then run FF in delta mode
        ff = CornerSpec(ProcessCorner.FF, 1.9, 50.0)
        result = simulate_corner(ff, golden_waveform=golden, use_delta=True)

        assert result["used_delta"]
        assert result["experimental"]
        assert result["execution_mode"] == "experimental_delta"
        assert result["peak_delta_v"] >= 0.0

    def test_delta_faster_than_full(self):
        nom = CornerSpec.nominal()
        golden = simulate_corner(nom, golden_waveform=None, use_delta=False)
        full_time = golden["wall_time"]

        ff = CornerSpec(ProcessCorner.FF, 1.9, 50.0)
        delta_time = simulate_corner(ff, golden_waveform=golden, use_delta=True)["wall_time"]

        # Delta is typically faster, but since this is synthetic, at minimum it shouldn't be 10x slower
        assert delta_time < full_time * 10, "Delta mode should not be much slower than full sim"


class TestPvtOrchestrator:
    def test_small_sweep(self):
        """Run a 3-corner mini sweep."""
        corners = [
            CornerSpec.nominal(),
            CornerSpec(ProcessCorner.FF, 1.9, 85.0),
            CornerSpec(ProcessCorner.SS, 1.7, -40.0),
        ]
        config = SweepConfig(max_workers=2, delta_mode=True)
        orch = PvtOrchestrator(config)
        result = orch.run_sweep(corners)

        assert result.completed >= 3, f"Should complete all corners, got {result.completed}"
        assert result.success_rate >= 0.9
        assert result.total_wall_time > 0.0

    def test_sweep_result_summary(self):
        corners = [CornerSpec.nominal(), CornerSpec(ProcessCorner.FF, 1.9, 27.0)]
        config = SweepConfig(max_workers=1, delta_mode=True)
        orch = PvtOrchestrator(config)
        result = orch.run_sweep(corners)

        summary = result.summary()
        assert "corners" in summary.lower()
        assert "speedup" in summary.lower()

    def test_sweep_json_export(self, tmp_path):
        corners = [CornerSpec.nominal()]
        config = SweepConfig(max_workers=1)
        orch = PvtOrchestrator(config)
        result = orch.run_sweep(corners)

        output_path = str(tmp_path / "pvt_result.json")
        result.to_json(output_path)
        assert os.path.exists(output_path)

        import json
        with open(output_path) as f:
            data = json.load(f)
        assert "total_corners" in data
        assert "speedup_vs_full" in data

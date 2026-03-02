"""
Phase 1 correctness harness tests.

These tests validate the initial correctness infrastructure:
  - Nangila emits full waveforms from the single-node solver
  - ngspice oracle execution works
  - waveform comparison metrics are computed
  - benchmark-specific behaviors like `.IC` handling are observable
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nangila_spice.correctness import (
    blocked_official_correctness_cases,
    compare_waveforms,
    extended_phase1_correctness_cases,
    find_nangila_binary,
    find_ngspice_binary,
    phase1_automated_correctness_cases,
    run_nangila_waveform,
    run_ngspice_waveform,
    within_v1_contract,
)


REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SIMPLE_RC = os.path.join(REPO_ROOT, "tests", "spice_examples", "simple_rc.sp")
INVERTER = os.path.join(REPO_ROOT, "benchmarks", "reference_circuits", "inverter.sp")
SRAM = os.path.join(REPO_ROOT, "benchmarks", "reference_circuits", "sram_6t.sp")
RUN_EXTENDED_CORRECTNESS = os.getenv("NANGILA_RUN_EXTENDED_CORRECTNESS") == "1"


class CorrectnessHarnessTests(unittest.TestCase):
    def setUp(self):
        if not find_nangila_binary():
            self.skipTest("nangila-node binary not found")

    def test_nangila_emits_full_waveform_for_simple_rc(self):
        waveform = run_nangila_waveform(SIMPLE_RC, tstop=1e-11, dt=1e-12)

        self.assertIn("vdd", waveform.node_names)
        self.assertIn("n1", waveform.node_names)
        self.assertGreaterEqual(len(waveform.times), 10)
        self.assertEqual(len(waveform.times), len(waveform.traces["n1"]))
        self.assertAlmostEqual(waveform.traces["vdd"][-1], 1.8, places=6)
        self.assertGreater(waveform.traces["n1"][-1], 1.0)

    def test_ngspice_oracle_runs_for_simple_rc(self):
        if not find_ngspice_binary():
            self.skipTest("ngspice binary not found")

        oracle = run_ngspice_waveform(
            SIMPLE_RC,
            nodes=["vdd", "n1"],
            tstop=1e-11,
            dt=1e-12,
        )

        self.assertGreaterEqual(len(oracle.times), 10)
        self.assertEqual(len(oracle.times), len(oracle.traces["n1"]))
        self.assertGreater(oracle.traces["n1"][-1], 1.0)

    def test_simple_rc_meets_v1_contract_tolerances(self):
        if not find_ngspice_binary():
            self.skipTest("ngspice binary not found")

        nangila = run_nangila_waveform(SIMPLE_RC, tstop=1e-11, dt=1e-12)
        oracle = run_ngspice_waveform(
            SIMPLE_RC,
            nodes=["vdd", "n1"],
            tstop=1e-11,
            dt=1e-12,
        )
        comparison = compare_waveforms(oracle, nangila, vdd=1.8)
        within_contract, profile = within_v1_contract(
            comparison,
            nonlinear=False,
            vdd=1.8,
        )

        self.assertGreater(comparison.sample_count, 0)
        self.assertTrue(math.isfinite(comparison.max_abs_error))
        self.assertTrue(math.isfinite(comparison.rms_error))
        self.assertTrue(within_contract)
        self.assertLessEqual(comparison.max_abs_error, profile.max_abs_tol)
        self.assertLessEqual(comparison.rms_error, profile.rms_tol)
        self.assertLessEqual(comparison.final_abs_error, profile.final_abs_tol)

    def test_inverter_waveform_has_real_transition(self):
        waveform = run_nangila_waveform(INVERTER, tstop=2e-9, dt=1e-11)

        in_trace = waveform.traces["in"]
        out_trace = waveform.traces["out"]
        self.assertGreater(max(in_trace) - min(in_trace), 1.7)
        self.assertGreater(max(out_trace) - min(out_trace), 0.7)

    def test_inverter_meets_v1_contract_tolerances(self):
        if not find_ngspice_binary():
            self.skipTest("ngspice binary not found")

        nangila = run_nangila_waveform(INVERTER, tstop=2e-9, dt=1e-11)
        oracle = run_ngspice_waveform(
            INVERTER,
            nodes=["in", "out", "vdd"],
            tstop=2e-9,
            dt=1e-11,
        )
        comparison = compare_waveforms(oracle, nangila, vdd=1.8)
        within_contract, profile = within_v1_contract(
            comparison,
            nonlinear=True,
            vdd=1.8,
        )

        self.assertTrue(within_contract)
        self.assertLessEqual(comparison.max_abs_error, profile.max_abs_tol)
        self.assertLessEqual(comparison.rms_error, profile.rms_tol)
        self.assertLessEqual(comparison.final_abs_error, profile.final_abs_tol)
        self.assertIsNotNone(comparison.max_edge_timing_error)
        self.assertLessEqual(comparison.max_edge_timing_error, profile.edge_tol)

    def test_sram_initial_conditions_are_visible_in_waveform(self):
        waveform = run_nangila_waveform(SRAM, tstop=2e-9, dt=5e-12)

        self.assertAlmostEqual(waveform.traces["q"][0], 1.8, places=6)
        self.assertAlmostEqual(waveform.traces["qb"][0], 0.0, places=6)
        self.assertGreater(min(waveform.traces["q"]) - max(waveform.traces["qb"]), 1.2)

    def test_sram_meets_v1_contract_tolerances(self):
        if not find_ngspice_binary():
            self.skipTest("ngspice binary not found")

        nangila = run_nangila_waveform(SRAM, tstop=5e-9, dt=1e-11)
        oracle = run_ngspice_waveform(
            SRAM,
            nodes=["q", "qb", "bl", "blb", "wl", "vdd"],
            tstop=5e-9,
            dt=1e-11,
            use_initial_conditions=True,
        )
        comparison = compare_waveforms(oracle, nangila, vdd=1.8)
        within_contract, profile = within_v1_contract(
            comparison,
            nonlinear=True,
            vdd=1.8,
        )

        self.assertTrue(within_contract)
        self.assertLessEqual(comparison.max_abs_error, profile.max_abs_tol)
        self.assertLessEqual(comparison.rms_error, profile.rms_tol)
        self.assertLessEqual(comparison.final_abs_error, profile.final_abs_tol)

    def test_phase1_automated_correctness_cases_meet_contract(self):
        if not find_ngspice_binary():
            self.skipTest("ngspice binary not found")

        cases = phase1_automated_correctness_cases(REPO_ROOT)
        self.assertIn("c432_auto", {case.name for case in cases})
        self.assertIn("c17_full", {case.name for case in cases})
        self.assertIn("c17_synth", {case.name for case in cases})
        self.assertIn("s27_auto", {case.name for case in cases})
        self.assertIn("s382_auto", {case.name for case in cases})
        self.assertIn("s641_auto", {case.name for case in cases})

        for case in cases:
            with self.subTest(case=case.name):
                nangila = run_nangila_waveform(
                    case.netlist_path,
                    tstop=case.tstop,
                    dt=case.dt,
                    vdd=case.vdd,
                )
                oracle = run_ngspice_waveform(
                    case.netlist_path,
                    nodes=case.nodes,
                    tstop=case.tstop,
                    dt=case.dt,
                    use_initial_conditions=case.use_initial_conditions,
                )
                comparison = compare_waveforms(oracle, nangila, vdd=case.vdd)
                within_contract, profile = within_v1_contract(
                    comparison,
                    nonlinear=case.nonlinear,
                    vdd=case.vdd,
                )

                self.assertTrue(within_contract)
                self.assertLessEqual(comparison.max_abs_error, profile.max_abs_tol)
                self.assertLessEqual(comparison.rms_error, profile.rms_tol)
                self.assertLessEqual(comparison.final_abs_error, profile.final_abs_tol)
                if profile.edge_tol is not None and comparison.max_edge_timing_error is not None:
                    self.assertLessEqual(comparison.max_edge_timing_error, profile.edge_tol)

    def test_blocked_official_correctness_cases_are_tracked(self):
        blocked = blocked_official_correctness_cases(REPO_ROOT)
        self.assertGreaterEqual(len(blocked), 1)
        self.assertNotIn("c432", {case.name for case in blocked})
        self.assertNotIn("c17_full", {case.name for case in blocked})
        self.assertNotIn("c17_synth", {case.name for case in blocked})
        self.assertNotIn("s27", {case.name for case in blocked})
        self.assertNotIn("s382", {case.name for case in blocked})
        self.assertNotIn("c1355", {case.name for case in blocked})
        self.assertNotIn("s641", {case.name for case in blocked})
        self.assertNotIn("c1908", {case.name for case in blocked})
        for case in blocked:
            with self.subTest(case=case.name):
                self.assertTrue(os.path.exists(case.netlist_path))
                self.assertTrue(case.reason)


@unittest.skipUnless(
    RUN_EXTENDED_CORRECTNESS,
    "Extended correctness cases are opt-in; set NANGILA_RUN_EXTENDED_CORRECTNESS=1 to run them.",
)
class ExtendedCorrectnessHarnessTests(unittest.TestCase):
    def setUp(self):
        if not find_nangila_binary():
            self.skipTest("nangila-node binary not found")
        if not find_ngspice_binary():
            self.skipTest("ngspice binary not found")

    def test_extended_phase1_cases_meet_contract(self):
        cases = extended_phase1_correctness_cases(REPO_ROOT)
        self.assertIn("c880_auto", {case.name for case in cases})
        self.assertIn("c1355_auto", {case.name for case in cases})
        self.assertIn("c1908_auto", {case.name for case in cases})

        for case in cases:
            with self.subTest(case=case.name):
                nangila = run_nangila_waveform(
                    case.netlist_path,
                    tstop=case.tstop,
                    dt=case.dt,
                    vdd=case.vdd,
                )
                oracle = run_ngspice_waveform(
                    case.netlist_path,
                    nodes=case.nodes,
                    tstop=case.tstop,
                    dt=case.dt,
                    use_initial_conditions=case.use_initial_conditions,
                )
                comparison = compare_waveforms(oracle, nangila, vdd=case.vdd)
                within_contract, profile = within_v1_contract(
                    comparison,
                    nonlinear=case.nonlinear,
                    vdd=case.vdd,
                )

                self.assertTrue(within_contract)
                self.assertLessEqual(comparison.max_abs_error, profile.max_abs_tol)
                self.assertLessEqual(comparison.rms_error, profile.rms_tol)
                self.assertLessEqual(comparison.final_abs_error, profile.final_abs_tol)


if __name__ == "__main__":
    unittest.main()

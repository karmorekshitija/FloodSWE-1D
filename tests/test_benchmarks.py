import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_hydraulics_sim_and_tests.py"
SPEC = importlib.util.spec_from_file_location("floodswe", MODULE_PATH)
floodswe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(floodswe)


class FloodSWEBenchmarkTests(unittest.TestCase):
    def test_stoker_profile_and_measured_shock(self):
        result = floodswe.run_saint_venant_dambreak_verification(num_cells=500)
        self.assertLess(result["l2_depth_error_vs_stoker_pct"], 2.5)
        self.assertLess(result["shock_speed_relative_error_pct"], 2.0)
        self.assertLessEqual(result["max_cfl_number"], 1.0)

    def test_grid_refinement_reduces_profile_error(self):
        results = floodswe.run_grid_convergence_study()
        errors = [row["l2_depth_error_pct"] for row in results]
        self.assertTrue(all(right < left for left, right in zip(errors, errors[1:])))

    def test_manning_calibration_is_labeled_synthetic(self):
        result = floodswe.run_manning_gvf_and_inverse_calibration()
        self.assertLess(result["manning_n_relative_error_pct"], 0.5)

    def test_adaptive_cfl_and_open_boundary_balance(self):
        result = floodswe.run_saint_venant_dambreak_verification(length_m=100, num_cells=60, t_end_s=25)
        self.assertLessEqual(result["max_cfl_number"], 0.45)
        self.assertLess(result["min_dt_s"], result["max_dt_s"])
        self.assertLess(result["volume_conservation_error_m3_per_m"], 1e-9)
        self.assertGreater(abs(result["raw_volume_change_m3_per_m"]), 1)
        self.assertFalse(result["exact_profile_valid_before_boundary_interaction"])

    def test_uniform_rest_and_invalid_depths(self):
        result = floodswe.run_saint_venant_dambreak_verification(num_cells=20, h_left=2, h_right=2)
        self.assertEqual(result["h_numerical_m"], [2.0]*20)
        self.assertFalse(result["shock_diagnostic_applicable"])
        for depth in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                floodswe.run_saint_venant_dambreak_verification(h_right=depth)

    def test_routing_budget_and_substep_sensitivity(self):
        coarse = floodswe.run_synthetic_storm_and_detention_scenario(12)
        fine = floodswe.run_synthetic_storm_and_detention_scenario(24)
        for result in (coarse, fine):
            self.assertLess(result["storage_balance_error_m3"], 1e-6)
        self.assertLess(abs(coarse["max_detention_basin_depth_m"]-fine["max_detention_basin_depth_m"]), 0.01)

    def test_gvf_rejects_critical_regime(self):
        from roughness_study import solve_profile
        with self.assertRaises(ValueError):
            solve_profile(0.016, downstream_depth=0.2)

    def test_roughness_recovery_and_sensor_validation(self):
        from roughness_study import estimate_roughness, solve_profile
        profile = solve_profile(0.018)
        indices = [30, 60, 120]
        estimate, residual = estimate_roughness(indices, [profile[index] for index in indices])
        self.assertAlmostEqual(estimate, 0.018, places=7)
        self.assertLess(residual, 1e-7)
        with self.assertRaises(ValueError):
            estimate_roughness([0], [2.6])


if __name__ == "__main__":
    unittest.main()

"""Subcritical GVF forward model and explicitly synthetic roughness experiments."""

import json
import math
import random
from pathlib import Path


def solve_profile(roughness, steps=150, slope=0.0012, discharge=42.0, width=8.0, length=1500.0, downstream_depth=2.6):
    values = (roughness, slope, discharge, width, length, downstream_depth)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("GVF parameters must be finite and positive")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    spacing = -length / steps

    def derivative(depth):
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError("GVF reached nonpositive or invalid depth")
        area = width * depth
        radius = area / (width + 2 * depth)
        velocity = discharge / area
        froude_squared = velocity * velocity / (9.81 * depth)
        if froude_squared >= 0.95:
            raise ValueError("Only subcritical GVF with Fr^2 < 0.95 is supported")
        friction = roughness ** 2 * velocity ** 2 / radius ** (4 / 3)
        return (slope - friction) / (1 - froude_squared)

    depth = downstream_depth
    profile = [depth]
    for _ in range(steps):
        first = derivative(depth)
        second = derivative(depth + spacing * first / 2)
        third = derivative(depth + spacing * second / 2)
        fourth = derivative(depth + spacing * third)
        depth += spacing * (first + 2 * second + 2 * third + fourth) / 6
        derivative(depth)
        profile.append(depth)
    return profile


def estimate_roughness(indices, observations, steps=150):
    if len(indices) != len(observations) or not indices or len(set(indices)) != len(indices):
        raise ValueError("Provide nonempty unique sensor indices and aligned observations")
    if any(not isinstance(index, int) or not 1 <= index <= steps for index in indices):
        raise ValueError("Sensors must lie upstream of the imposed downstream boundary")
    if any(not math.isfinite(value) or value <= 0 for value in observations):
        raise ValueError("Observed depths must be finite and positive")

    def loss(roughness):
        profile = solve_profile(roughness, steps=steps)
        return sum((profile[index] - observation) ** 2 for index, observation in zip(indices, observations))

    lower, upper = 0.010, 0.035
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        left = upper - ratio * (upper-lower)
        right = lower + ratio * (upper-lower)
        if loss(left) < loss(right):
            upper = right
        else:
            lower = left
    estimate = (lower+upper) / 2
    return estimate, math.sqrt(loss(estimate) / len(indices))


def run_study():
    results = []
    for slope_factor in (1.0, 1.1):
        truth = solve_profile(0.016, steps=600, slope=0.0012*slope_factor)
        for sensor_layout, indices in (("distributed", [15, 45, 90, 120, 150]), ("near_boundary", [1, 2, 3, 4, 5])):
            for noise_m in (0.0, 0.002, 0.01):
                for seed in range(4):
                    rng = random.Random(seed)
                    observations = [truth[4*index] + rng.gauss(0, noise_m) for index in indices]
                    estimate, observation_rmse = estimate_roughness(indices, observations)
                    reconstructed = solve_profile(estimate)
                    truth_rmse = math.sqrt(sum((value-truth[4*index])**2 for index, value in enumerate(reconstructed)) / len(reconstructed))
                    results.append({
                        "seed": seed, "noise_std_m": noise_m, "sensor_layout": sensor_layout,
                        "true_slope_factor": slope_factor, "assumed_slope_factor": 1.0,
                        "true_n": 0.016, "estimated_n": estimate,
                        "signed_n_error_pct": 100*(estimate/0.016-1),
                        "observation_fit_rmse_m": observation_rmse,
                        "true_profile_rmse_m": truth_rmse,
                    })
    report = {"scope": "48 synthetic cases; finer forward grid and optional 10% slope mismatch; no field validation",
              "truth_steps": 600, "inverse_steps": 150, "cases": results}
    path = Path(__file__).resolve().parent / "reports" / "roughness_uncertainty.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(f"Completed {len(run_study()['cases'])} synthetic roughness cases")

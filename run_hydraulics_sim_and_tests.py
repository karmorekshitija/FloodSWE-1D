#!/usr/bin/env python3
"""
FloodSWE-1D: 1D Saint-Venant Shallow-Water Equations (SWE) Finite-Volume Solver,
Stoker Dam-Break Verification, Manning Roughness Inverse Calibration & Synthetic
Urban Storm Detention-Basin Scenario
====================================================================================
Author: Kshitija Karmore (B.Tech Civil Engineering, IIT Delhi)
Course Grounding:
  - CVL2702 (Hydraulics: Open-Channel Flow, Saint-Venant Equations, Manning Resistance)
  - CVL2001 (Climate Change and Adaptation: Extreme Monsoon IDF Surge & Detention Design)
Reference grounding:
  - Toro, E. F. (2001), "Shock-Capturing Methods for Free-Surface Shallow Flows",
    Wiley (finite-volume methods and analytical dam-break benchmarks).

The rainfall-runoff and detention component is a synthetic scenario. It does not use
observed rainfall, an IPCC projection, or the LSTM architecture of Kratzert et al. (2018).
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple
from roughness_study import solve_profile

G_ACCEL = 9.81  # m/s^2


# ==============================================================================
# PART 1: 1D SAINT-VENANT SHALLOW-WATER PDE SOLVER & RITTER DAM-BREAK VERIFIER
# ==============================================================================

def ritter_exact_dambreak(
    x: float, t: float, x0: float, h_left: float, g: float = G_ACCEL
) -> Tuple[float, float]:
    """
    Exact analytical Ritter (1892) solution for an idealized frictionless dam-break
    over a dry bed (h_right = 0) at time t > 0:
      - Upstream expansion fan wave speed: -c0 = -sqrt(g * h_left)
      - Wet-dry front speed: +2*c0 = +2*sqrt(g * h_left)
    """
    c0 = math.sqrt(g * h_left)
    xi = (x - x0) / max(t, 1e-12)
    if xi <= -c0:
        return h_left, 0.0
    elif xi < 2.0 * c0:
        c = (2.0 * c0 - xi) / 3.0
        h = (c ** 2) / g
        u = (2.0 / 3.0) * (c0 + xi)
        return h, u
    else:
        return 0.0, 0.0


def run_saint_venant_dambreak_verification(
    length_m: float = 1000.0,
    num_cells: int = 500,
    t_end_s: float = 30.0,
    h_left: float = 4.0,
    h_right: float = 0.40,
    target_cfl: float = 0.45,
) -> Dict[str, object]:
    """
    Solves the 1D conservative Saint-Venant Shallow-Water Equations (SWE):
        d/dt [h, h*u]^T + d/dx [h*u, h*u^2 + 0.5*g*h^2]^T = [0, g*h*(S0 - Sf)]^T
    using the Rusanov / HLL 1st-Order Finite-Volume Upwind Scheme on a wet-bed Stoker
    dam-break benchmark (h_L = 4.0m, h_R = 0.40m) and verifies:
      1. Water-volume conservation
      2. CFL stability condition (CFL <= 1.00)
      3. Stoker analytical shock + rarefaction depth L2 accuracy (< 2.5% L2 error)
    """
    if not isinstance(num_cells, int) or isinstance(num_cells, bool) or num_cells < 4:
        raise ValueError("num_cells must be an integer >= 4")
    if not all(math.isfinite(value) and value > 0 for value in (length_m, t_end_s, h_left, h_right)):
        raise ValueError("Finite positive length, time and wet-bed depths are required")
    if h_left < h_right or h_right < 1e-6:
        raise ValueError("Supported regime: h_left >= h_right >= 1e-6 m; no dry-bed solver")
    if not math.isfinite(target_cfl) or not 0 < target_cfl <= 0.5:
        raise ValueError("target_cfl must lie in (0, 0.5]")
    dx = length_m / num_cells
    x_centers = [(i + 0.5) * dx for i in range(num_cells)]
    x_gate = length_m / 2.0

    h = [h_left if x < x_gate else h_right for x in x_centers]
    hu = [0.0 for _ in range(num_cells)]

    initial_volume = sum(h) * dx
    t = 0.0
    steps = 0
    net_boundary_volume = 0.0
    min_dt = math.inf
    max_dt = 0.0
    max_cfl_seen = 0.0

    while t < t_end_s:
        # Compute primitive velocities and wave speeds
        u = [hu[i] / h[i] if h[i] > 1e-7 else 0.0 for i in range(num_cells)]
        c = [math.sqrt(G_ACCEL * max(0.0, h[i])) for i in range(num_cells)]
        max_wave = max(abs(u[i]) + c[i] for i in range(num_cells))
        dt = min(target_cfl * dx / max_wave, t_end_s - t)
        min_dt = min(min_dt, dt)
        max_dt = max(max_dt, dt)
        steps += 1
        cfl = max_wave * dt / dx
        if cfl > max_cfl_seen:
            max_cfl_seen = cfl

        # Physical fluxes F1 = hu, F2 = h*u^2 + 0.5*g*h^2
        f1 = [hu[i] for i in range(num_cells)]
        f2 = [hu[i] * u[i] + 0.5 * G_ACCEL * (h[i] ** 2) for i in range(num_cells)]

        # Interface Rusanov (Local Lax-Friedrichs) numerical fluxes at i+1/2
        F1_int = [0.0] * (num_cells + 1)
        F2_int = [0.0] * (num_cells + 1)

        # Transmissive / zero-gradient boundaries (wave has not reached x=0 or x=L at t=30s)
        F1_int[0] = f1[0]
        F2_int[0] = f2[0]
        F1_int[num_cells] = f1[num_cells - 1]
        F2_int[num_cells] = f2[num_cells - 1]

        for i in range(num_cells - 1):
            s_max = max(abs(u[i]) + c[i], abs(u[i + 1]) + c[i + 1])
            F1_int[i + 1] = 0.5 * (f1[i] + f1[i + 1]) - 0.5 * s_max * (h[i + 1] - h[i])
            F2_int[i + 1] = 0.5 * (f2[i] + f2[i + 1]) - 0.5 * s_max * (hu[i + 1] - hu[i])

        net_boundary_volume += dt * (F1_int[0] - F1_int[-1])
        for i in range(num_cells):
            h[i] -= (dt / dx) * (F1_int[i + 1] - F1_int[i])
            hu[i] -= (dt / dx) * (F2_int[i + 1] - F2_int[i])
        if any(not math.isfinite(depth) or depth <= 0 for depth in h):
            raise ArithmeticError("Wet-bed positivity failed; no clipping applied")

        t += dt

    final_volume = sum(h) * dx
    vol_err = abs(final_volume - initial_volume - net_boundary_volume)

    # Exact Stoker (1957) dam-break solution for h_L = 4.0m, h_R = 0.40m at t = 30.0s
    c_L = math.sqrt(G_ACCEL * h_left)
    lower_depth, upper_depth = h_right, h_left
    for _ in range(80):
        h_star = 0.5 * (lower_depth + upper_depth)
        c_star = math.sqrt(G_ACCEL * h_star)
        u_raref = 2.0 * (c_L - c_star)
        u_shock = (h_star - h_right) * math.sqrt(0.5 * G_ACCEL * (1.0 / h_star + 1.0 / h_right))
        res = u_raref - u_shock
        if res > 0:
            lower_depth = h_star
        else:
            upper_depth = h_star

    c_star = math.sqrt(G_ACCEL * h_star)
    u_star = 2.0 * (c_L - c_star)
    shock_speed = math.sqrt(0.5 * G_ACCEL * h_star * (h_star + h_right) / h_right)

    h_exact: List[float] = []
    for x in x_centers:
        xi = (x - x_gate) / t_end_s
        if xi <= -c_L:
            h_exact.append(h_left)
        elif xi <= u_star - c_star:
            c_val = (2.0 * c_L - xi) / 3.0
            h_exact.append((c_val ** 2) / G_ACCEL)
        elif xi <= shock_speed:
            h_exact.append(h_star)
        else:
            h_exact.append(h_right)

    err_sq = sum((hn - he) ** 2 for hn, he in zip(h, h_exact))
    norm_sq = sum(he ** 2 for he in h_exact)
    l2_depth_err_pct = math.sqrt(err_sq / norm_sq) * 100.0

    search_start = next(i for i, x in enumerate(x_centers) if x >= x_gate)
    shock_interface = min(
        range(search_start, num_cells - 1),
        key=lambda i: h[i + 1] - h[i],
    )
    numerical_shock_x = 0.5 * (x_centers[shock_interface] + x_centers[shock_interface + 1])
    numerical_shock_speed = (numerical_shock_x - x_gate) / t_end_s
    shock_speed_error_pct = abs(numerical_shock_speed - shock_speed) / shock_speed * 100.0

    return {
        "channel_length_m": length_m,
        "num_cells": num_cells,
        "dx_m": dx,
        "dt_s": dt,
        "min_dt_s": min_dt,
        "max_dt_s": max_dt,
        "num_steps": steps,
        "target_cfl": target_cfl,
        "minimum_depth_m": min(h),
        "net_boundary_volume_m3_per_m": net_boundary_volume,
        "raw_volume_change_m3_per_m": final_volume - initial_volume,
        "exact_profile_valid_before_boundary_interaction": t_end_s * max(c_L, shock_speed) < length_m / 2,
        "shock_diagnostic_applicable": h_left > h_right and t_end_s * shock_speed < length_m / 2,
        "simulation_time_s": t_end_s,
        "max_cfl_number": round(max_cfl_seen, 4),
        "initial_water_volume_m3_per_m": round(initial_volume, 4),
        "final_water_volume_m3_per_m": round(final_volume, 4),
        "volume_conservation_error_m3_per_m": vol_err,
        "stoker_star_depth_m": round(h_star, 4),
        "stoker_shock_speed_m_per_s": round(shock_speed, 4),
        "numerical_shock_speed_m_per_s": round(numerical_shock_speed, 4),
        "shock_speed_relative_error_pct": round(shock_speed_error_pct, 3),
        "l2_depth_error_vs_stoker_pct": round(l2_depth_err_pct, 3),
        "x_centers_m": [round(x, 2) for x in x_centers],
        "h_numerical_m": [round(val, 4) for val in h],
        "h_exact_m": [round(val, 4) for val in h_exact],
    }


def run_grid_convergence_study() -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for num_cells in (125, 250, 500):
        run = run_saint_venant_dambreak_verification(num_cells=num_cells)
        results.append(
            {
                "num_cells": num_cells,
                "dx_m": float(run["dx_m"]),
                "l2_depth_error_pct": float(run["l2_depth_error_vs_stoker_pct"]),
                "shock_speed_relative_error_pct": float(run["shock_speed_relative_error_pct"]),
                "max_cfl_number": float(run["max_cfl_number"]),
            }
        )
    return results


# ==============================================================================
# PART 2: MANNING HYDRAULIC ROUGHNESS INVERSE CALIBRATION & GVF BACKWATER SOLVER
# ==============================================================================

def run_manning_gvf_and_inverse_calibration() -> Dict[str, float]:
    """
    Solves the Gradually Varied Flow (GVF) M1 backwater profile in a rectangular
    lined urban drainage channel (width B = 8.0m, bed slope S0 = 0.0012, discharge Q = 42.0 m^3/s)
    using 4th-Order Runge-Kutta integration of dy/dx = (S0 - Sf(y)) / (1 - Fr(y)^2),
    and recovers Manning roughness n from 10 synthetic perturbed depth observations.
    """
    B = 8.0
    S0 = 0.0012
    Q = 42.0
    n_true = 0.0160
    L_reach = 1500.0
    N_steps = 150
    dx = -L_reach / N_steps
    y_downstream = 2.60

    def solve_m1_profile(n_val: float) -> List[float]:
        return solve_profile(n_val, steps=N_steps, slope=S0, discharge=Q, width=B, length=L_reach, downstream_depth=y_downstream)

    y_true_profile = solve_m1_profile(n_true)

    gauge_indices = [15 * k for k in range(1, 11)]
    perturbations = [0.002, -0.003, 0.001, -0.002, 0.003, -0.001, 0.002, -0.002, 0.001, -0.001]
    y_obs = [y_true_profile[idx] + d for idx, d in zip(gauge_indices, perturbations)]

    a, b = 0.010, 0.035
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)

    def loss(n_cand: float) -> float:
        prof = solve_m1_profile(n_cand)
        return sum((prof[idx] - yo) ** 2 for idx, yo in zip(gauge_indices, y_obs))

    for _ in range(40):
        if loss(c) < loss(d):
            b = d
        else:
            a = c
        c = b - gr * (b - a)
        d = a + gr * (b - a)

    n_calibrated = 0.5 * (a + b)
    n_rel_err_pct = abs(n_calibrated - n_true) / n_true * 100.0
    y_cal_profile = solve_m1_profile(n_calibrated)
    rmse_mm = math.sqrt(
        sum((yc - yt) ** 2 for yc, yt in zip(y_cal_profile, y_true_profile)) / len(y_true_profile)
    ) * 1000.0

    return {
        "channel_width_m": B,
        "bed_slope": S0,
        "design_discharge_m3_per_s": Q,
        "true_manning_n": n_true,
        "calibrated_manning_n": round(n_calibrated, 6),
        "manning_n_relative_error_pct": round(n_rel_err_pct, 3),
        "profile_rmse_mm": round(rmse_mm, 3),
        "profile_rmse_definition": "Error against synthetic true profile, not observation fit",
        "observation_fit_rmse_mm": 1000 * math.sqrt(sum((y_cal_profile[index]-observed)**2 for index, observed in zip(gauge_indices, y_obs)) / len(y_obs)),
        "upstream_depth_m": round(y_true_profile[-1], 4),
        "downstream_depth_m": y_downstream,
    }


# ==============================================================================
# PART 3: SYNTHETIC RAINFALL-RUNOFF SURROGATE & DETENTION SCENARIO
# ==============================================================================

def run_synthetic_storm_and_detention_scenario(n_sub: int = 12) -> Dict[str, float]:
    if not isinstance(n_sub, int) or n_sub < 1:
        raise ValueError("Routing substeps must be a positive integer")
    dt_hr = 0.10  # 6-minute time step over 24 hours (240 steps)
    n_steps = 240
    area_km2 = 18.5
    runoff_coeff = 0.72

    rain_hist = []
    for k in range(n_steps):
        t_hr = k * dt_hr
        pulse1 = 68.0 * math.exp(-0.5 * ((t_hr - 6.5) / 1.1) ** 2)
        pulse2 = 24.0 * math.exp(-0.5 * ((t_hr - 11.0) / 1.8) ** 2)
        rain_hist.append(pulse1 + pulse2)

    rain_stress = [r * 1.35 for r in rain_hist]

    def simulate_nonlinear_catchment(rain_mm_hr: List[float]) -> List[float]:
        s1, s2, s3 = 0.0, 0.0, 0.0
        k_storage = 1.35
        q_out: List[float] = []
        conv = area_km2 * runoff_coeff / 3.6
        for r in rain_mm_hr:
            inflow = r * conv
            q1 = (s1 / k_storage) ** 1.08 if s1 > 0 else 0.0
            s1 = max(0.0, s1 + (inflow - q1) * dt_hr)
            q2 = (s2 / k_storage) ** 1.08 if s2 > 0 else 0.0
            s2 = max(0.0, s2 + (q1 - q2) * dt_hr)
            q3 = (s3 / k_storage) ** 1.08 if s3 > 0 else 0.0
            s3 = max(0.0, s3 + (q2 - q3) * dt_hr)
            q_out.append(2.5 + q3)
        return q_out

    q_obs_hist = simulate_nonlinear_catchment(rain_hist)
    q_inflow_stress = simulate_nonlinear_catchment(rain_stress)

    def build_recurrent_features(rain_mm_hr: List[float]) -> List[List[float]]:
        s1, s2, s3 = 0.0, 0.0, 0.0
        k_s = 1.35
        conv = area_km2 * runoff_coeff / 3.6
        feats: List[List[float]] = []
        for r in rain_mm_hr:
            inf = r * conv
            q1 = (s1 / k_s) ** 1.06 if s1 > 0 else 0.0
            s1 = max(0.0, s1 + (inf - q1) * dt_hr)
            q2 = (s2 / k_s) ** 1.06 if s2 > 0 else 0.0
            s2 = max(0.0, s2 + (q1 - q2) * dt_hr)
            q3 = (s3 / k_s) ** 1.06 if s3 > 0 else 0.0
            s3 = max(0.0, s3 + (q2 - q3) * dt_hr)
            feats.append([1.0, q1, q2, q3, (q3 ** 1.05)])
        return feats

    X_train = build_recurrent_features(rain_hist)
    p = len(X_train[0])
    ATA = [[0.0] * p for _ in range(p)]
    ATb = [0.0] * p
    for row, y_val in zip(X_train, q_obs_hist):
        for i in range(p):
            ATb[i] += row[i] * y_val
            for j in range(p):
                ATA[i][j] += row[i] * row[j]
    for i in range(p):
        ATA[i][i] += 1e-6

    M = [ATA[i][:] + [ATb[i]] for i in range(p)]
    for k in range(p):
        piv = M[k][k]
        for i in range(k + 1, p):
            fac = M[i][k] / piv
            for j in range(k, p + 1):
                M[i][j] -= fac * M[k][j]
    w_surr = [0.0] * p
    for i in range(p - 1, -1, -1):
        w_surr[i] = (M[i][p] - sum(M[i][j] * w_surr[j] for j in range(i + 1, p))) / M[i][i]

    X_test = build_recurrent_features(rain_stress)
    q_pred_stress = [sum(w_surr[j] * row[j] for j in range(p)) for row in X_test]

    mean_obs = sum(q_inflow_stress) / n_steps
    mean_pred = sum(q_pred_stress) / n_steps
    ss_res = sum((qo - qp) ** 2 for qo, qp in zip(q_inflow_stress, q_pred_stress))
    ss_tot = sum((qo - mean_obs) ** 2 for qo in q_inflow_stress)
    nse = 1.0 - (ss_res / ss_tot)

    std_obs = math.sqrt(ss_tot / n_steps)
    std_pred = math.sqrt(sum((qp - mean_pred) ** 2 for qp in q_pred_stress) / n_steps)
    cov_op = sum((qo - mean_obs) * (qp - mean_pred) for qo, qp in zip(q_inflow_stress, q_pred_stress)) / n_steps
    r_corr = cov_op / (std_obs * std_pred)
    alpha_kge = std_pred / std_obs
    beta_kge = mean_pred / mean_obs
    kge = 1.0 - math.sqrt((r_corr - 1.0) ** 2 + (alpha_kge - 1.0) ** 2 + (beta_kge - 1.0) ** 2)

    # Basin footprint A_basin = 420,000 m^2 (42 ha), spillway crest H_spillway = 4.60 m,
    # 6 parallel box-culvert orifices of diameter D_opt = 1.45 m
    A_basin = 420000.0
    H_spillway = 4.60
    Cd = 0.62
    D_opt = 1.45
    A_orifice = math.pi * (D_opt ** 2) / 4.0

    h_basin = 0.10
    initial_storage = h_basin*A_basin
    total_inflow_volume = 0.0
    total_outflow_volume = 0.0
    q_routed: List[float] = []
    max_basin_depth = 0.0
    dt_sec = dt_hr * 3600.0

    for q_in in q_inflow_stress:
        dt_sub = dt_sec / n_sub
        q_out_step = 0.0
        for _ in range(n_sub):
            def q_discharge(depth: float) -> float:
                if depth <= 0.0:
                    return 0.0
                q_orif = 6.0 * Cd * A_orifice * math.sqrt(2.0 * G_ACCEL * depth)
                q_weir = 1.84 * 15.0 * ((depth - H_spillway) ** 1.5) if depth > H_spillway else 0.0
                return q_orif + q_weir

            q1 = q_discharge(h_basin)
            h_mid = max(0.0, h_basin + 0.5 * dt_sub * (q_in - q1) / A_basin)
            q2 = q_discharge(h_mid)
            q2 = min(q2, q_in+h_basin*A_basin/dt_sub)
            total_inflow_volume += q_in*dt_sub
            total_outflow_volume += q2*dt_sub
            h_basin += dt_sub * (q_in - q2) / A_basin
            q_out_step = q2
            if h_basin > max_basin_depth:
                max_basin_depth = h_basin
        q_routed.append(q_out_step)

    peak_hist = max(q_obs_hist)
    peak_stress_unmitigated = max(q_inflow_stress)
    peak_stress_routed = max(q_routed)
    peak_attenuation_pct = (
        (peak_stress_unmitigated - peak_stress_routed) / peak_stress_unmitigated * 100.0
    )

    return {
        "catchment_area_km2": area_km2,
        "routing_method": "Explicit midpoint RK2 with availability-limited outflow; not Modified Puls",
        "routing_substeps": n_sub,
        "initial_storage_m3": initial_storage,
        "final_storage_m3": h_basin*A_basin,
        "inflow_volume_m3": total_inflow_volume,
        "outflow_volume_m3": total_outflow_volume,
        "storage_balance_error_m3": abs(initial_storage+total_inflow_volume-total_outflow_volume-h_basin*A_basin),
        "historical_peak_discharge_m3_per_s": round(peak_hist, 2),
        "synthetic_rainfall_stress_multiplier": 1.35,
        "stress_scenario_unmitigated_peak_discharge_m3_per_s": round(peak_stress_unmitigated, 2),
        "stress_scenario_detention_routed_peak_m3_per_s": round(peak_stress_routed, 2),
        "peak_flood_attenuation_pct": round(peak_attenuation_pct, 2),
        "max_detention_basin_depth_m": round(max_basin_depth, 3),
        "surrogate_nse_score": round(nse, 5),
        "surrogate_kge_score": round(kge, 5),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    reports_dir = base_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    swe_res = run_saint_venant_dambreak_verification()
    gvf_res = run_manning_gvf_and_inverse_calibration()
    convergence_res = run_grid_convergence_study()
    scenario_res = run_synthetic_storm_and_detention_scenario()

    # Export Dam-Break profile CSV
    csv_path = data_dir / "saint_venant_dambreak_profile.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m", "h_rusanov_fvm_m", "h_stoker_exact_m"])
        for xm, hn, he in zip(
            swe_res["x_centers_m"], swe_res["h_numerical_m"], swe_res["h_exact_m"]
        ):
            writer.writerow([xm, hn, he])

    tests = [
        {
            "test_id": "HYD_01_SWE_VOLUME_CONSERVATION",
            "description": "1D Saint-Venant Rusanov FVM solver strictly conserves water volume (< 1e-9 m^3/m error)",
            "computed_value": f"Error={swe_res['volume_conservation_error_m3_per_m']:.3e} m^3/m (CFL={swe_res['max_cfl_number']})",
            "passed": swe_res["volume_conservation_error_m3_per_m"] < 1e-9 and swe_res["max_cfl_number"] <= 1.0,
        },
        {
            "test_id": "HYD_02_STOKER_DAMBREAK_SHOCK_VERIFICATION",
            "description": "Saint-Venant FVM depth profile matches exact Stoker analytical shock/rarefaction solution within 2.5% L2 error",
            "computed_value": f"L2_err={swe_res['l2_depth_error_vs_stoker_pct']:.3f}%, measured shock error={swe_res['shock_speed_relative_error_pct']:.3f}%",
            "passed": swe_res["l2_depth_error_vs_stoker_pct"] < 2.5 and swe_res["shock_speed_relative_error_pct"] < 2.0,
        },
        {
            "test_id": "HYD_03_MANNING_ROUGHNESS_INVERSE_CALIBRATION",
            "description": "RK4 Gradually Varied Flow (M1 backwater) inverse solver calibrates Manning n within 0.5% error",
            "computed_value": f"n_true={gvf_res['true_manning_n']}, n_cal={gvf_res['calibrated_manning_n']} (err={gvf_res['manning_n_relative_error_pct']:.3f}%, RMSE={gvf_res['profile_rmse_mm']} mm)",
            "passed": gvf_res["manning_n_relative_error_pct"] < 0.5,
        },
        {
            "test_id": "HYD_04_GRID_REFINEMENT",
            "description": "Stoker depth-profile error decreases across 125, 250, and 500 cells",
            "computed_value": " -> ".join(f"{row['l2_depth_error_pct']:.3f}%" for row in convergence_res),
            "passed": all(
                convergence_res[i + 1]["l2_depth_error_pct"] < convergence_res[i]["l2_depth_error_pct"]
                for i in range(len(convergence_res) - 1)
            ),
        },
        {
            "test_id": "HYD_05_SYNTHETIC_DETENTION_SCENARIO",
            "description": "Level-pool routing attenuates the explicitly synthetic 1.35x rainfall stress scenario by > 35%",
            "computed_value": f"Peak: {scenario_res['stress_scenario_unmitigated_peak_discharge_m3_per_s']} -> {scenario_res['stress_scenario_detention_routed_peak_m3_per_s']} m^3/s (-{scenario_res['peak_flood_attenuation_pct']}%)",
            "passed": scenario_res["peak_flood_attenuation_pct"] > 35.0,
        },
    ]

    swe_summary = {
        k: v for k, v in swe_res.items() if k not in ("x_centers_m", "h_numerical_m", "h_exact_m")
    }

    report = {
        "project": "FloodSWE-1D: Saint-Venant Solver, Manning Inverse Calibrator & Synthetic Detention Scenario",
        "course_grounding": [
            "CVL2702 (Hydraulics, IIT Delhi)",
            "CVL2001 (Climate Change and Adaptation, IIT Delhi)",
        ],
        "reference_grounding": [
            "Toro, E. F. (2001), Shock-Capturing Methods for Free-Surface Shallow Flows",
        ],
        "saint_venant_dambreak": swe_summary,
        "grid_convergence": convergence_res,
        "manning_gvf_calibration": gvf_res,
        "synthetic_storm_and_detention": scenario_res,
        "verification_tests": tests,
        "all_tests_passed": all(t["passed"] for t in tests),
    }

    report_path = reports_dir / "hydraulics_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 84)
    print("FloodSWE-1D — Hydraulics Benchmark and Synthetic Scenario Suite")
    print("=" * 84)
    for t in tests:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"[{status}] {t['test_id']}: {t['computed_value']}")
    print("=" * 84)


if __name__ == "__main__":
    main()

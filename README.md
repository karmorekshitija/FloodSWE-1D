# FloodSWE-1D

A zero-dependency computational hydraulics project containing an analytical dam-break benchmark, a synthetic Manning-roughness inverse exercise, and a hypothetical detention-basin routing scenario.

## Evidence boundary

The wet-bed Stoker dam-break comparison is the externally defined analytical benchmark. The Manning calibration and rainfall-runoff components use data generated inside the script. This repository does not reproduce an LSTM, use observed flood records, or apply an IPCC regional projection. See `METHODOLOGY.md`.

## 1. Saint-Venant benchmark

The solver integrates the conservative one-dimensional shallow-water equations with a first-order Rusanov finite-volume flux:

$$
\frac{\partial}{\partial t}
\begin{bmatrix}h\\hu\end{bmatrix}
+
\frac{\partial}{\partial x}
\begin{bmatrix}hu\\hu^2+\frac{1}{2}gh^2\end{bmatrix}
=0.
$$

Analytical verification includes:

- full-profile $L_2$ error against the analytical wet-bed Stoker solution;
- shock speed measured from the numerical depth gradient and compared with the analytical speed;
- water-volume conservation and CFL checks;
- a 125, 250, and 500-cell grid-refinement study.
- current-state adaptive CFL selection, end-time clipping and boundary-flux mass balance;
- explicit wet-bed input restrictions and rest-state/open-boundary regression checks.

Reference grounding: Toro, E. F. (2001), *Shock-Capturing Methods for Free-Surface Shallow Flows*.

## 2. Synthetic inverse exercise

The script solves an M1 backwater profile with RK4 and recovers a known Manning coefficient from ten perturbed depths. Because the observations come from the same model family, this demonstrates the mechanics of inverse calibration but not field accuracy.

`roughness_study.py` adds 48 cases across four seeds, three noise levels, two sensor layouts and optional 10% slope mismatch. Truth uses a finer grid than estimation. Critical/supercritical GVF is rejected instead of silently clipping the governing denominator. Observation-fit error and error against synthetic truth are reported separately.

## 3. Synthetic detention scenario

An invented 24-hour hyetograph is multiplied by 1.35 for stress testing. A synthetic nonlinear catchment model generates inflow, and a hypothetical 42-hectare basin is routed with a sub-stepped level-pool calculation. NSE, KGE, and peak attenuation are implementation diagnostics for this constructed case only.

## Repository structure

```text
flood_swe_hydraulics_simulator/
|-- run_hydraulics_sim_and_tests.py
|-- METHODOLOGY.md
|-- tests/test_benchmarks.py
|-- data/saint_venant_dambreak_profile.csv
`-- reports/hydraulics_benchmark_report.json
```

## Run

```bash
python run_hydraulics_sim_and_tests.py
python roughness_study.py
python -m unittest discover -s tests -v
```

The main command regenerates the CSV profile and JSON report.

Read `reports/RESEARCH_REPORT.md` for measured results, negative cases and limitations. Run the local `build_research_reports.py` to regenerate three SVG figures. Level-pool routing uses explicit midpoint RK2, not Modified Puls, and includes a storage budget. Student explanation and reproduction remain pending; implementation is AI-assisted.

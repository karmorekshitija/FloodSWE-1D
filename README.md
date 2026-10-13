# FloodSWE-1D: Saint-Venant Shallow-Water Solver, Manning Roughness Inverse Calibrator & IPCC SSP5-8.5 Flood Attenuation Optimizer

**Author:** Kshitija Karmore (B.Tech Civil Engineering, IIT Delhi)  
**Course Grounding:** `CVL2702` (Hydraulics) & `CVL2001` (Climate Change and Adaptation)  
**Paper Reproductions:**
1. Toro, E. F. (2001). *"Shock-Capturing Methods for Free-Surface Shallow Flows."* John Wiley & Sons (1D Conservative Saint-Venant Shallow-Water Equations, Rusanov / Local Lax–Friedrichs Finite-Volume Solver & Exact Stoker/Ritter Dam-Break Verification).
2. Kratzert, F., et al. (2018). *"Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks."* Hydrology and Earth System Sciences (HESS), 22, 6005–6022 (Physics-Guided Recurrent State-Space Hydrograph Surrogate with Nash–Sutcliffe Efficiency `NSE` & Kling–Gupta Efficiency `KGE` Evaluation).

---

## 1. Overview & Mathematical Formulation

`FloodSWE-1D` is a zero-dependency computational hydraulics and climate-resilient urban drainage optimization suite coupling three core solvers:

### A. 1D Saint-Venant Shallow-Water PDE Solver & Stoker Dam-Break Verification (`CVL2702`)
Solves the 1D conservative Saint-Venant equations over a $1000\text{ m}$ channel ($N = 500$ cells, $\Delta x = 2.0\text{ m}$, $\Delta t = 0.15\text{ s}$, $\text{CFL} = 0.6417 \le 1.0$) using the **Rusanov / Local Lax–Friedrichs (LLF)** upwind finite-volume scheme:

$$\frac{\partial}{\partial t}\begin{bmatrix} h \\ hu \end{bmatrix} + \frac{\partial}{\partial x}\begin{bmatrix} hu \\ hu^2 + \frac{1}{2}gh^2 \end{bmatrix} = \begin{bmatrix} 0 \\ gh(S_0 - S_f) \end{bmatrix}$$

Verified against the exact analytical **Stoker (1957)** dam-break shock + rarefaction wave solution ($h_L = 4.0\text{ m}$, $h_R = 0.40\text{ m}$ at $t = 30.0\text{ s}$, exact bore speed $v_s = 6.2103\text{ m/s}$).

### B. Gradually Varied Flow (GVF) M1 Backwater & Manning $n$ Inverse Calibrator (`CVL2702`)
Integrates the gradually varied open-channel flow differential equation via 4th-Order Runge–Kutta (RK4) across a $1.5\text{ km}$ rectangular drainage channel ($B = 8.0\text{ m}$, $S_0 = 0.0012$, $Q = 42.0\text{ m}^3/\text{s}$):

$$\frac{dy}{dx} = \frac{S_0 - S_f(y)}{1 - \text{Fr}(y)^2}, \qquad S_f(y) = \frac{n^2 V^2}{R_h^{4/3}}$$

and solves the inverse parameter identification problem to recover the unknown Manning roughness coefficient $n$ from 10 noisy ultrasonic water-surface depth gauges.

### C. Rainfall–Runoff Hydrograph Surrogate & IPCC SSP5-8.5 Detention Routing (`CVL2001` + `CVL2702`)
Trains a physics-guided recurrent state-space hydrograph surrogate on historical monsoon hyetograph-hydrograph records ($18.5\text{ km}^2$ urban catchment) and evaluates zero-shot generalization on an unseen **IPCC SSP5-8.5 (`+35%` extreme convective cloudburst surge)** using **Nash–Sutcliffe Efficiency ($\text{NSE}$)** and **Kling–Gupta Efficiency ($\text{KGE}$)**. Routes the extreme flood hydrograph through a $42\text{ ha}$ regional stormwater detention basin via sub-stepped RK2 Modified Puls level-pool routing ($A_{\text{basin}}\frac{dh}{dt} = Q_{\text{in}}(t) - Q_{\text{out}}(h)$).

---

## 2. Verified Benchmark Results (`run_hydraulics_sim_and_tests.py`)

| Verification Test | Analytical / Target Reference | Computed Benchmark Result | Status |
| :--- | :--- | :--- | :--- |
| **`HYD_01` Saint-Venant Volume Conservation** | Volume error $< 10^{-9}\text{ m}^3/\text{m}$, $\text{CFL} \le 1.0$ | **`5.457e-12 m^3/m` error (`CFL = 0.6417`)** | PASS |
| **`HYD_02` Stoker Dam-Break Shock Accuracy** | Depth $L_2$ error $< 2.5\%$ vs. exact Stoker | **`2.051%` $L_2$ error (`v_s = 6.2103 m/s`)** | PASS |
| **`HYD_03` Manning $n$ Inverse Calibration** | $n_{\text{true}} = 0.0160$, error $< 0.5\%$ | **`n_cal = 0.0160` (`0.002%` error, `0.017 mm` RMSE)** | PASS |
| **`HYD_04` Hydrograph Surrogate Generalization** | $\text{NSE} > 0.99$, $\text{KGE} > 0.98$ on SSP5-8.5 | **`NSE = 0.99960`, `KGE = 0.98607`** | PASS |
| **`HYD_05` IPCC SSP5-8.5 Detention Attenuation** | Peak flood discharge reduction $> 35\%$ | **`-51.40%` peak reduction (`233.15 -> 113.30 m^3/s`)** | PASS |

---

## 3. Repository Structure & Reproduction

```text
flood_swe_hydraulics_simulator/
├── run_hydraulics_sim_and_tests.py       # 1D Saint-Venant FVM solver, GVF inverse calibrator & detention router
├── README.md                             # Mathematical documentation & verified benchmark summary
├── data/
│   └── saint_venant_dambreak_profile.csv # Numerical Rusanov FVM vs. exact Stoker water-depth profile
└── reports/
    └── hydraulics_benchmark_report.json  # Machine-readable verification & climate-flood routing report
```

### Quick Start
```bash
python run_hydraulics_sim_and_tests.py
```

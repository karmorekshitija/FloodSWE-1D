# Hydraulics research report

## Question and method
How do discretization and imperfect observations affect wet-bed flow verification and roughness recovery? A first-order Rusanov solver uses current-state characteristic speeds to select every timestep (target CFL 0.45), clips the last step to the end time, and accumulates boundary mass flux. Supported inputs require h_left >= h_right >= 1e-6 m. Dry beds and critical/supercritical GVF are explicitly unsupported rather than silently clipped.

## Numerical evidence
The 1 km, 4 m/0.4 m wet-bed dam break at 30 seconds has 2.053% relative depth-profile error at 500 cells and 1.239% measured shock-speed error. Profile errors at 125/250/500 cells are 4.615%, 3.176%, 2.053%. Maximum CFL is 0.45; the mass-balance residual including boundary flux is 4.09e-12 m3 per metre width. Open-boundary and rest-state regression cases also pass. The infinite-domain analytical comparison is flagged inapplicable after waves reach boundaries.

## Roughness sensitivity
The original near-exact recovery is only one same-family synthetic exercise. The new study contains 48 cases: four seeds, three noise levels, two sensor layouts and two slope assumptions. Truth uses 600 integration steps; fitting uses 150. The optional 10% slope mismatch is deliberately not fitted. Absolute roughness error ranges from 1.054e-07% to 22.78%. Reported observation-fit RMSE and synthetic true-profile RMSE are distinct. Four seeds are a sensitivity check, not a calibrated confidence interval or proof of identifiability. Poor sensor placement and slope uncertainty can dominate measurement noise.

## Supporting detention exercise
An invented rainfall stress scenario uses midpoint RK2 level-pool routing, not Modified Puls. Outflow is capped by available storage and inflow. The saved budget residual is 8.85e-09 m3. No observed storm, climate projection or engineering design adequacy is established.

## Evidence and reproduction
Reference equations and wet-bed wave structure: [Clawpack Riemann book](https://www.clawpack.org/riemann_book/html/Shallow_water.html); classic analytical benchmark: Stoker, as discussed in Toro (2001). This is analytical verification, not an independent experimental dataset or externally executed solver comparison.

Run `python -B run_hydraulics_sim_and_tests.py`, `python -B roughness_study.py` and `python -B -m unittest discover -s tests -q`. Generate figures with the `build_research_reports.py` in this folder. Inputs and all 48 outcomes are retained in `roughness_uncertainty.json`. Figures: `dam_break_profile.svg`, `roughness_uncertainty.svg`, `roughness_mismatch.svg`.

Agent-assisted implementation; student explanation and reproduction remain to be demonstrated. No field-validation CV wording is supported.

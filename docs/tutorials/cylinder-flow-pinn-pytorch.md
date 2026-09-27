---
title: Low-Reynolds-number cylinder flow with a PINN
short_title: Cylinder flow
description: Explore steady Re=20 cylinder flow with a physics-only PINN and inspect residuals and mass balance without simulation-derived reference data.
scenario: cylinder
topic: Incompressible flow
equation: '\nabla\cdot\mathbf{u}=0,\quad (\mathbf{u}\cdot\nabla)\mathbf{u}+\nabla p=Re^{-1}\Delta\mathbf{u}'
---

# Flow around a cylinder

**No simulation data:** neither training nor evaluation may use numerical
reference fields or published simulation-derived drag, lift, or pressure values.
This scenario is **exploratory and accuracy-unverified**: no exact reference is
selected for this channel-cylinder setup. Follow the
[staged validation plan](../validation.md), including exact circular Couette
flow before attempting to assess curved-wall problems.

The `cylinder` scenario solves the steady, incompressible Navier–Stokes equations
with three network outputs, `(u, v, p)`. Validate these coupled equations first
with `kovasznay`, whose exact interior velocity and pressure are known.

## Domain and units

This case follows the [DFG 2D-1 benchmark documented by FEATFLOW](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html).
Its dimensional channel is `[0, 2.2] × [0, 0.41]`, with a radius-0.05 cylinder
centered at `(0.2, 0.2)`. Density is 1, viscosity is 0.001, and mean inlet speed
is 0.2. Scaling length by cylinder diameter `D=0.1`, speed by `U=0.2`, and
pressure by `rho*U²=0.04` gives the computational domain
`[0, 22] × [0, 4.1]`, center `(2, 2)`, radius `0.5`, and `Re=20`.

The residuals are

$$
r_c=u_x+v_y,\qquad
r_u=uu_x+vu_y+p_x-Re^{-1}(u_{xx}+u_{yy}),\qquad
r_v=uv_x+vv_y+p_y-Re^{-1}(v_{xx}+v_{yy}).
$$

Only second spatial derivatives are required. The smooth network has four
hidden layers of 64 tanh units.

## Boundary conditions

- Inlet: `u=6*y*(4.1-y)/4.1²`, `v=0` (mean speed 1).
- Channel walls and cylinder surface: `u=v=0`.
- Outlet: `u_x/Re-p=0`, `v_x/Re=0`.

The outlet stress condition fixes the additive pressure constant. Do not also
impose an unrelated pressure value. Kovasznay instead fixes pressure at one
corner because it prescribes velocities on all edges.

A coordinate-aware output transform builds the cylinder's inlet and no-slip
conditions into the prediction. The transform uses a smooth circle-distance
factor and the prescribed inlet parabola; it supplies no exact interior flow.
Pressure remains an independent network output. Boundary errors are still
measured and logged.

## Train and inspect

```bash
uv run learnpdes train kovasznay --epochs 1500 --lbfgs-steps 1500 --points 31 --no-gif
uv run learnpdes train circular-couette --epochs 1500 --lbfgs-steps 2000 --points 32 --no-gif
uv run learnpdes train cylinder --epochs 1500 --lbfgs-steps 2500 --points 45 --no-gif
uv run learnpdes refine path/to/completed-cylinder-run --points 81 --lbfgs-steps 3000
uv run learnpdes plot path/to/refined-cylinder-run --resolution 601 --png
```

For these fluid cases, `--points N` selects `N²` interior points and `4*N`
independent points per boundary. Cylinder training draws interior
points across the channel (40%), in the upstream obstacle region (20%), near
the inlet (20%), and in an annulus around the cylinder (20%). Scrambled Sobol
sampling reduces gaps in the rectangular regions. Solid and boundary points are excluded from PDE
sampling. `--resample-every 100` refreshes samples during Adam; all coordinates
remain fixed throughout L-BFGS, including line-search evaluations. Every loss
evaluation creates fresh coordinate graphs. `--threads 1` is the CPU default.

Each run saves weights, metadata, total loss, separate unweighted PDE/boundary
MSEs in `residuals.csv`, and an interactive training figure with a cylinder hole.
The optimized loss sums the PDE MSEs and ten times each boundary MSE. It also
enforces integral continuity across 12 channel sections, using the prescribed
inlet flow rate as the target (weight 10, logged as `mass_flux`). Quadrature
excludes the solid cylinder. This adds no exact interior velocity or pressure
data. Inlet refinement and integral continuity prevent low-flow fits that can
hide continuity defects between collocation points. Remove
`--no-gif` to export a GIF as well. Display resolution does not change training
or validation samples.

The [problem definitions and objective](../../learnpdes/scenarios/cylinder/problem.py),
[trainer](../../learnpdes/model/trainer.py), and
[independent evaluator](../../learnpdes/scenarios/cylinder/evaluation.py) have separate roles.
The evaluator reports predictions and physics diagnostics only for this case.
Each fluid run also saves a separate `verification.json`, including reference
provenance, two independent sampling budgets, per-check pass/fail, and the
explicit distinction between single-run checks and project acceptance.
See the [measured scenario log and plots](../fluid-scenario-log.md).

## Inspect physics without a simulation reference

There is no exact interior solution supplied for this cylinder channel.
FEATFLOW's high-order spectral drag, lift, and pressure-drop values were
computed by numerical simulation. They are prohibited even as held-out
comparison targets and are not used by the evaluator.

Drag, lift, and front-to-back pressure drop remain **predicted observables**,
with no reference-error metric. Pressure is sampled at `(1.5, 2)` and `(2.5, 2)`;
surface quadrature computes forces using normals pointing out of the solid.
No agreement with a known cylinder solution is claimed.

The evaluator also reports each PDE RMS on unseen uniformly sampled points and
independently sampled boundary errors. Independent quadrature checks flux
against the prescribed inlet flow rate, **4.1 in model units**, and inlet/outlet
mass balance. This flux follows analytically by integrating the prescribed
inlet profile; it is not a simulation result. Conservation and small residuals
are necessary diagnostics, not a bound on force or full-field accuracy.

Use Kovasznay for exact full-field velocity/pressure comparisons without fitting
a pressure offset. Use the [circular Couette case](circular-couette-pinn-pytorch.md) for an exact curved-wall
problem; it does not supply reference values for this different cylinder setup.

Run the fast equation, geometry, boundary, derivative, and optimizer checks:

```bash
uv run python -m unittest tests.scenarios.test_cylinder tests.scenarios.test_kovasznay -v
```

Run the longer training regressions explicitly:

```bash
LEARNPDES_FLUID_ACCURACY=1 uv run python -m unittest tests.scenarios.test_fluid_convergence -v
```

These existing checks use seed 0 for training and 4,096 evaluation points
from a different seed. The Kovasznay regression targets below 1% relative L2
error for each output and PDE RMS below 0.02. The cylinder regression checks
PDE RMS below 0.01 after dense refinement and outlet flux within 1% of the analytically prescribed
inlet flux. It does not check drag/lift/pressure accuracy.

These are partial checks, not the stronger proposed three-seed acceptance gate
in the [validation plan](../validation.md). Cylinder remains unverified even
when its diagnostic checks pass.

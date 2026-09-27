---
title: Circular Couette flow with a PINN
short_title: Circular Couette
description: Verify curved no-slip walls against an exact annular Navier–Stokes solution before progressing to obstacle flow.
scenario: circular-couette
topic: Incompressible flow
equation: 'u_\theta=\frac23(r-r^{-1}),\quad p_r=u_\theta^2/r'
---

# Circular Couette: verify curved walls first

This separate test scenario adds curved walls to the steady incompressible
Navier–Stokes model. It has an exact analytical reference, unlike the channel
cylinder and bundled airfoil problems. No simulation data enters training or
evaluation. The broader [validation plan](../validation.md) still governs
promotion to harder cases.

The fluid occupies the annulus `1 < r < 2`, with viscosity `0.1` (`Re=10`).
The inner wall is stationary. The outer wall rotates with tangential speed 1,
so its Cartesian velocity is `(-y/2, x/2)`. Pressure is fixed once, at `(2,0)`:
`p=0`. These dimensionless parameters are chosen for this test.

The independently implemented exact reference is

$$
u_\theta=\frac23(r-r^{-1}),\qquad
(u,v)=\frac23(1-r^{-2})(-y,x),
$$

$$
p(r)=\frac{2r^2-8\log r-2r^{-2}}9-
      \frac{8-8\log2-1/2}9.
$$

This follows from the azimuthal viscous equation and radial balance
`p_r=u_theta²/r`; the [Couette and Poiseuille notes from MIT](https://ocw.mit.edu/courses/2-25-advanced-fluid-mechanics-fall-2013/resources/mit2_25f13_couet_and_pois/)
introduce the viscous-flow setting. Our formula, radii, speed, and pressure
constant are stated explicitly here. At `r=1.5`, `u_theta=5/9`; at `r=2`,
`u_theta=1` and `p=0`.

The network predicts `(u,v,p)` with four 64-unit tanh layers and the same
continuity and momentum residuals as Kovasznay/cylinder. A polynomial wall
extension enforces the prescribed velocities exactly. It does not insert the
rational exact interior solution. Interior points are uniform in annular area;
inner/outer boundary points are sampled separately. The analytical evaluator
is never called by the training objective.

```bash
uv run learnpdes train circular-couette --epochs 1500 --lbfgs-steps 2000 --points 32 --no-gif
uv run python -m unittest tests.scenarios.test_circular_couette -v
```

Each run has its own weights, residual CSV, interactive prediction/reference/
error plots, configuration, and `verification.json`. Verification uses 4,096
and 8,192 independently sampled interior points, separate boundary points,
per-output errors, and individual PDE residuals. A passing single run is logged
separately from the stronger three-seed, two-density project acceptance gate.

Measured runs and the next staged cases are listed in the
[fluid scenario log](../fluid-scenario-log.md).

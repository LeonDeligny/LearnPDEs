---
title: Solve plane Poiseuille flow with a PINN in PyTorch
short_title: Poiseuille flow
description: Solve steady viscous channel flow with a PyTorch PINN. Enforce no-slip walls and a pressure drop, build Navier–Stokes residuals, and check the parabolic profile.
scenario: poiseuille
topic: Incompressible flow
equation: '\mu u_{yy} = p_x'
---

# Poiseuille flow

**No simulation data:** train only on the equations, prescribed pressures,
and wall/end conditions; compare only with the analytical profile below.
Exact interior values never enter training. Centerline speed is 1 and flux is
$2/3$. Use only the analytical formula from cited sources, never their numerical
solver results. See the [staged validation gate](../validation.md); this case's
vanishing convection cannot establish nonlinear transport accuracy.

## Problem

Find horizontal velocity $u$, vertical velocity $v$, and pressure $p$ in the
channel $[0,4]\times[0,1]$. The fluid is steady and incompressible, with density
$\rho=1$, dynamic viscosity $\mu=0.1$, and no body force. All values use
dimensionless model units. The network predicts $(x,y)\mapsto(u,v,p)$ and
satisfies the full two-dimensional Navier–Stokes equations:

$$
\begin{aligned}
u_x+v_y &= 0,\\
\rho(u u_x+v u_y)+p_x-\mu(u_{xx}+u_{yy}) &= 0,\\
\rho(u v_x+v v_y)+p_y-\mu(v_{xx}+v_{yy}) &= 0.
\end{aligned}
$$

Prescribe the walls and the two open ends:

| Boundary | Conditions |
| --- | --- |
| Plates, $y=0$ and $y=1$ | $u=0$, $v=0$ (no slip and no penetration) |
| Inlet, $x=0$ | $p=3.2$, $u_x=0$, $v=0$ |
| Outlet, $x=4$ | $p=0$, $u_x=0$, $v=0$ |

The pressure drop drives the flow. The end conditions describe fully developed
flow and fix the pressure reference; they do not prescribe a velocity profile.
For this solution, the momentum equation reduces to $\mu u_{yy}=p_x$. With
$G=-p_x=3.2/4=0.8$, the exact solution is

$$
u(y)=\frac{G}{2\mu}y(1-y)=4y(1-y),\qquad
v=0,\qquad p(x)=0.8(4-x).
$$

This parabolic profile follows the pressure–viscosity balance described in the
[MFiX plane Poiseuille verification case](https://mfix.netl.doe.gov/doc/vvuq-manual/main/html/fluid/fld-01.html).
The constants and open-end conditions above are the LearnPDEs setup.

## Run

After [setup](README.md#install), run:

```bash
uv run learnpdes train poiseuille --epochs 5000 --points 21
```

This samples a $21\times21$ training grid, including the boundaries. The exact
velocity and pressure fields are used only for evaluation. The command prints
its run folder, `assets/runs/poiseuille/<run-id>/`. Open `training.gif` or
`training.html` there; PNG checkpoints are in `frames/`, and `run.json` records
settings and validation results. Use `--output-dir` for another output root
or `--no-gif` for HTML without Chrome or FFmpeg.

![Poiseuille PINN training between stationary parallel plates](../../assets/examples/poiseuille/training.gif)

The figure shows both velocity components, pressure, and training loss. The
interactive version also offers reference and signed-error views. This is a
recorded example; new runs leave it unchanged. Export a new interactive run with
`uv run learnpdes train poiseuille --no-gif`.

## Loss

Let $r_c$, $r_u$, and $r_v$ be the continuity and two momentum residuals above.
Minimize their mean squared values and the boundary penalties:

$$
\mathcal L=3\left(\operatorname{mean}[r_c^2]
+\operatorname{mean}[r_u^2]+\operatorname{mean}[r_v^2]\right)
+\mathcal L_{\mathrm{walls}}+\mathcal L_{\mathrm{inlet}}
+\mathcal L_{\mathrm{outlet}}.
$$

The wall loss penalizes $u$ and $v$. Each end loss penalizes its pressure error,
$u_x$, and $v$, with a separate mean squared term for each condition.

Save as `inspect_loss.py` in the repo root; run `uv run python inspect_loss.py`.

```python
import torch
from learnpdes.training import build_problem
from learnpdes import poiseuille

torch.manual_seed(0)
model, problem, _ = build_problem('poiseuille', points=21)
u, v, p = model(problem.inputs).split(1, dim=1)
derivative = problem.partial_derivative
ux, uy = derivative(u, problem.x), derivative(u, problem.y)
vx, vy = derivative(v, problem.x), derivative(v, problem.y)
px, py = derivative(p, problem.x), derivative(p, problem.y)
lap_u = derivative(ux, problem.x) + derivative(uy, problem.y)
lap_v = derivative(vx, problem.x) + derivative(vy, problem.y)

residuals = (
    ux + vy,
    problem.rho * (u * ux + v * uy) + px - poiseuille.VISCOSITY * lap_u,
    problem.rho * (u * vx + v * vy) + py - poiseuille.VISCOSITY * lap_v,
)
physics = sum(residual.square().mean() for residual in residuals)
boundary = u[problem.wall_mask].square().mean() + v[problem.wall_mask].square().mean()
for mask, pressure in (
    (problem.inlet_mask, poiseuille.INLET_PRESSURE),
    (problem.outlet_mask, poiseuille.OUTLET_PRESSURE),
):
    boundary = (
        boundary
        + (p[mask] - pressure).square().mean()
        + ux[mask].square().mean()
        + v[mask].square().mean()
    )
loss = 3 * physics + boundary

reference, *_ = problem.poiseuille_loss()
torch.testing.assert_close(loss, reference)
print('Physics:', physics.item(), 'Boundary:', boundary.item())
```

Keep gradients enabled for the first and second derivatives. Splitting the
network output into `(N, 1)` columns preserves the shapes used by the boundary
masks. All conditions are soft penalties. The loss learns the velocity profile
from pressure forcing, viscosity, and the walls; it never compares training
velocities with the parabola. [Loss source](../../learnpdes/scenarios/poiseuille.py).

## Check

The runner reports aggregate and per-component errors on a separate
$41\times41$ grid. Check for a symmetric velocity profile with zero wall speed,
centerline speed $u(1/2)=1$, and pressure decreasing linearly from $3.2$ to $0$.
The vertical velocity should approach zero everywhere. Its exact field has
zero norm, so the runner reports absolute errors for $v$ rather than relative
L2 error.

A zero velocity field satisfies continuity and the wall conditions but fails
the pressure-driven momentum balance. Inspect all three fields and their
boundary errors; low total loss alone does not establish accuracy. Compare
`--epochs`, `--points`, and `--seed`. Display resolution (`--resolution`) only
changes the visualization grid.

[Training source](../../learnpdes/training.py) ·
[Reference solution](../../learnpdes/scenarios/poiseuille.py) ·
[Benchmark tests](../../tests/scenarios/test_poiseuille.py)

---
title: Validation without simulation data
short_title: Validation plan
description: Progress from exact ODEs to PDEs and viscous flow using published analytical solutions, reproducible comparison values, and explicit acceptance gates without simulation data.
---

# Validation without simulation data

**Never use simulation data in any scenario, for training, tuning, validation,
or comparison.** This includes numerical ODE trajectories, CFD/FEM/FDM/spectral
fields, another PINN's predictions, simulation-derived boundary data, and even
a single drag or lift value computed by another solver and published in a paper.

Training uses equations, known coefficients/forcing, geometry, and prescribed
initial/boundary conditions only. Coordinate samples and geometry-only meshes
are allowed. Analytical boundary/initial traces define a problem; analytical
interior values are evaluation-only. They must not drive supervised losses,
pretraining, residual weights, adaptive sampling, or early stopping. Formula
evaluation, automatic differentiation, and quadrature of our own predictions
do not import simulation data.

Our references are closed-form analytical solutions, or explicitly labelled
manufactured exact solutions with a derived forcing term. Mathematical checks
verify the solution of the stated equations; they do not establish agreement
with physical experiments. No experimental comparison dataset is selected.

## Current scenarios and admissible comparisons

Implemented does not mean accepted. None of the new multi-seed acceptance
requirements below is claimed complete by this plan.

| Runnable scenario | No simulation data: allowed reference | Current limitation |
| --- | --- | --- |
| `exponential` | Analytical $e^x$ only | Re-establish the multi-seed baseline on disjoint samples. |
| `forced-linear` | Analytical $e^{-t/5}\sin t$ only (Lagaris problem 2) | Runnable; full multi-seed/two-density gate remains open. |
| `logistic` | Analytical $(1+e^{-t})^{-1}$ only | Runnable; full multi-seed/two-density gate remains open. |
| `cosinus` | Analytical $\cos x$ and its derivatives only | Accept order 2 first; higher orders are derivative stress tests. |
| `laplace` | Analytical $\sin(\pi x)\sinh(\pi y)/\sinh\pi$ only | Existing evaluation and training grids overlap. |
| `poiseuille` | Analytical $u=4y(1-y)$, $v=0$, $p=0.8(4-x)$ only | Convection vanishes; this cannot verify nonlinear transport. |
| `kovasznay` | Kovasznay's exact velocity and pressure only | Current opt-in convergence test uses one training seed. |
| `wind_tunnel_no_geometry` | Uniform analytical velocity $(1,0)$ only | No independent velocity accuracy evaluator yet; audit pressure sign/gauge. |
| `circular-couette` | Exact annular Couette formula; pressure gauge at outer wall | Single-seed checks pass; multi-seed/two-density gate remains open. |
| `cylinder` | No admissible solution reference selected; no simulation data | Exploratory. Predicted forces, flux balance, and residuals are diagnostics, not drag/lift accuracy evidence. |
| `potential-flow` | No admissible solution reference selected; no simulation data | Exploratory airfoil geometry; mesh coordinates/connectivity are not solution labels. |
| `solenoidal-flow` | No admissible solution reference selected; no simulation data | Exploratory airfoil streamfunction; zero pressure output is a placeholder, not a validated pressure field. |

The DFG cylinder's published reference forces/pressure were obtained numerically
and are excluded. The [FEATFLOW benchmark](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)
may document geometry and equations; its computed reference tables must not
supply comparison targets. Historical run exports are not evidence of compliance
with this new gate and must not be relabelled as accepted results.

## Acceptance gate for every step

These are **proposed project targets**, not paper-reported tolerances or measured
results. Keep the gate fixed before a run. If a case fails, investigate the same
case; do not advance, relax thresholds, or select a favorable seed.

1. Record the equation, domain/time interval, coefficients, units/scales,
   boundary/initial data, pressure gauge, source URL and equation/section, exact
   formula, and reference type (`analytical`, `manufactured`, or `none`).
   Identify which constants were chosen here instead of copied from a paper.
2. Independently differentiate the formula and substitute into every residual
   and condition in float64. Target normalized residuals below $10^{-10}$ for
   these smooth cases. Check hand-derived point values as well, so training and
   evaluation cannot share an unnoticed sign or scaling error.
3. Freeze model, optimizer, sample counts, normalization, and training budget.
   Train with seeds **0, 1, 2**, recording all outcomes. Exact interior values
   never enter optimization or checkpoint selection. Separate development
   reference comparisons from final acceptance evaluation; changing the setup
   requires a fresh recorded acceptance run.
4. Use disjoint held-out coordinates: at least 1,024 for ODEs/1D stationary
   problems, 4,096 for 2D or space–time problems, and 256 per boundary/initial
   surface. Reserve evaluation seeds 1729 and 2027. Remove coordinate overlap;
   a denser training grid is not wholly unseen. Report edge, corner,
   pressure-anchor, and initial-condition checks separately.
5. For **every nonzero output**, require relative L2 error <= 1% and maximum
   absolute error / declared characteristic output scale <= 2%. For identically
   zero outputs or a zero-norm time slice, use absolute RMSE / scale <= 0.001 and
   maximum error / scale <= 0.01. Never divide by a near-zero reference norm.
   Use scale 1 for unit-amplitude cases, $e^3$ for exponential growth on
   $[-3,3]$, and 3.2 for Poiseuille pressure. Record separate velocity/pressure
   scales; an aggregate metric cannot conceal a failed component.
6. Require each dimensionless PDE RMS <= 0.01 and soft BC/IC RMS <= 0.001,
   using fixed characteristic equation scales derived from the stated units,
   not the measured residual. Hard BCs must hold to floating-point accuracy.
   Check conserved flux/energy, or its analytical decay law, within 1% where
   applicable. No post-hoc pressure-offset fitting: use the prescribed gauge.
7. Double the held-out sample budget and surface quadrature; both evaluations
   must pass. Repeat training at a higher collocation density with the same
   declared optimizer budget and all three seeds; both densities must pass.
   For time-dependent cases, report slices at $t=0,T/4,T/2,3T/4,T$, with errors
   and invariants at each slice. For Burgers also inspect points near its layer.
8. Save configuration, source revision, seed, dtype/device, formula provenance,
   individual residuals, metrics, thresholds, and pass/fail in a reviewable run
   report. Promote only when **all seeds** pass. A smoke test, attractive figure,
   or low training loss alone cannot pass this gate.

Current evaluators do not yet implement this full gate. In particular, the
single-seed fluid regressions and overlapping rectangular grids are partial
checks. Adding the gate is the first implementation task.

## Progression and exact comparison values

Every step below inherits **no simulation data** and the gate above. Values are
rounded evaluations of the displayed formulas, not PINN results or copied
numerical solver output. Recompute at full precision; table rounding is not an
accuracy tolerance. Source labels link to papers or primary teaching sources
under [Sources](#sources). Elementary cases are derived here; method papers are
not being credited with our parameter choices or decimal values.

Only the registered scenarios in the first table are currently runnable. Other names
are proposals, not CLI options. First accept A, then B. C and the steady fluid
branch D1–D4 can follow B separately. D5 requires C1 and D3; D6 requires C3,
C5, and D5. Change one equation feature or parameter at a time.

### A. Ordinary differential equations

A1–A3 are configured as `exponential`, `forced-linear`, and `logistic`;
see the [training commands and defaults](tutorials/README.md#first-order-ode-progression-a1a3).

| Step | Defined problem and exact reference | Comparison values and purpose | Source |
| --- | --- | --- | --- |
| A1 — exponential | $f'=f$, $f(0)=1$, $x\in[-3,3]$; $f=e^x$ | $f(-1)=0.367879441171$, $f(0)=1$, $f(1)=2.718281828459$. First derivative and initial value. | Elementary solution; method [L]. |
| A2 — forced linear | $f'+f/5=e^{-t/5}\cos t$, $f(0)=0$, $t\in[0,2]$; $f=e^{-t/5}\sin t$ | $f(1)=0.688938173085$, $f(2)=0.609520293010$. Introduce forcing. | [L], section 4.1.2, problem 2. |
| A3 — logistic | $f'=f(1-f)$, $f(0)=1/2$, $t\in[0,2]$; $f=(1+e^{-t})^{-1}$ | $f(1)=0.731058578630$, $f(2)=0.880797077978$; require $0<f<1$ and monotonicity. | Elementary separable ODE; method [L]. |
| A4 — harmonic oscillator | $f''+f=0$, $f(0)=1$, $f'(0)=0$, $t\in[-\pi,\pi]$; $f=\cos t$ | $f(\pi/2)=0$, $f(\pi)=-1$; energy $(f'^2+f^2)/2=1/2$. Add one derivative. | Elementary solution; method [L], section 3.1. |
| A5 — damped oscillator | $f''+0.4f'+1.04f=0$, $f(0)=1$, $f'(0)=-0.2$, $t\in[0,2\pi]$; $f=e^{-t/5}\cos t$ | $f(\pi)=-0.533488091091$, $f(2\pi)=0.284609543336$; check phase and decay separately. | Project-defined exact case; method [L]. |
| A6 — coupled oscillator | $q'=v$, $v'=-q$, $(q,v)(0)=(1,0)$ on $[0,2\pi]$; $(q,v)=(\cos t,-\sin t)$ | At $t=\pi/2$: $(0,-1)$; $q^2+v^2=1$. Two coupled outputs with unchanged physics. | Reformulation of A4; method [L], section 3.1. |

A5 intentionally uses initial derivative $-0.2$, not zero. Higher even derivative
constraints on `cosinus` wait until A4 passes; they are a differentiation stress
test, not a substitute for additional well-posed ODEs/PDEs.

### B. Stationary scalar PDEs

| Step | Defined problem and exact reference | Comparison values and purpose | Source |
| --- | --- | --- | --- |
| B1 — 1D Poisson | $-u''=2$ on $[0,1]$, $u(0)=u(1)=0$; $u=x(1-x)$ | $u(1/4)=3/16$, $u(1/2)=1/4$. Move to a boundary-value problem. | Project-defined manufactured case; method [M]. |
| B2 — Laplace | $\Delta u=0$ on $[0,1]^2$; top $u=\sin\pi x$, other edges zero; $u=\sin(\pi x)\sinh(\pi y)/\sinh\pi$ | Center $0.199268407669$; top center 1. Second spatial coordinate. | Separation of variables; boundary method [L]. |
| B3 — 2D Poisson | $-\Delta u=2\pi^2\sin(\pi x)\sin(\pi y)$ on $[0,1]^2$, zero Dirichlet edges; $u=\sin(\pi x)\sin(\pi y)$ | Center 1; $(1/4,1/2)$: $1/\sqrt2$. Add a nonzero source. | Project-defined manufactured case; method [M]. |
| B4 — mixed Poisson | Same equation/reference as B3; replace top Dirichlet with $u_y(x,1)=-\pi\sin(\pi x)$; other edges remain Dirichlet | Center 1; outward normal derivative at top center $-\pi$. Test normals and derivative conditions. | Project-defined case; methods [M], [L] section 3.2. |

Derive manufactured forcing analytically before training. It is part of the
equation; the selected interior solution is not a label. For B2, compare soft
penalties with the boundary-only transform
$u_\theta=y\sin(\pi x)+x(1-x)y(1-y)N_\theta(x,y)$ after the baseline passes.
Keep seeds and budgets identical when changing boundary treatment.

### C. Time-dependent scalar PDEs

These separable/travelling solutions and normalizations are derived here.
Initial and boundary traces are prescribed problem data only.

| Step | Defined problem and exact reference | Comparison values and purpose | Source |
| --- | --- | --- | --- |
| C1 — heat | $u_t=0.1u_{xx}$, $x,t\in[0,1]$; zero endpoints, $u(x,0)=\sin\pi x$; $u=e^{-0.1\pi^2t}\sin\pi x$ | $u(1/2,1/2)=0.610498025266$, $u(1/2,1)=0.372707838853$. Add time and an initial surface. | Derived Fourier mode; verification method [M]. |
| C2 — advection | $u_t+u_x=0$ on periodic $x\in[0,2\pi]$, $t\in[0,1]$, $u(x,0)=\sin x$; $u=\sin(x-t)$ | $u(\pi/2+1,1)=1$; spatial mean 0 and mean square $1/2$. Check phase, periodicity, transport. | Derived characteristic solution; method [M]. |
| C3 — advection–diffusion | $u_t+u_x=0.1u_{xx}$; C2 domain/IC, periodic values and first spatial derivatives; $u=e^{-0.1t}\sin(x-t)$ | $u(\pi/2+1,1)=0.904837418036$; mean square $e^{-0.2t}/2$. Combine accepted operators. | Derived Fourier mode; method [M]. |
| C4 — wave | $u_{tt}=u_{xx}$, $x,t\in[0,1]$; zero endpoints, $u(x,0)=\sin\pi x$, $u_t(x,0)=0$; $u=\sin\pi x\cos\pi t$ | Center: 1 at $t=0$, 0 at $t=1/2$, -1 at $t=1$; energy $\int_0^1(u_t^2+u_x^2)/2\,dx=\pi^2/4$. | Derived standing wave; method [M]. |
| C5 — viscous Burgers | $u_t+uu_x=\nu u_{xx}$, $x\in[-1,1]$, $t\in[0,1]$; $\nu=0.1$, $u=c-a\tanh[a(x-ct)/(2\nu)]$, $a=c=1/2$; prescribe its traces at $t=0,x=\pm1$ | At $t=0$: $u(-1)=0.993307149076$, $u(0)=1/2$, $u(1)=0.006692850924$; at $t=1$: $u(1/2)=1/2$. Smooth nonlinear transport. | Analytical Burgers theory [B]; constants chosen here. |

Start Burgers at $\nu=0.1$, then consider 0.05 and 0.01 as separate gates.
Never load a conventional numerically generated Burgers `.mat` benchmark.
The wave's zero solution slice needs absolute errors, not relative L2.

### D. Incompressible flow

Density is 1 throughout. For steady flow use continuity and
$(\mathbf u\cdot\nabla)\mathbf u+\nabla p-\nu\Delta\mathbf u=0$;
unsteady cases add $\mathbf u_t$. Pressure is kinematic pressure in these units.
Use direct $(u,v,p)$ outputs and check each momentum residual separately.

| Step | Defined setup and exact reference | Comparison values and purpose | Source |
| --- | --- | --- | --- |
| D1 — plane Couette | Unit square, $\nu=0.1$, periodic in $x$, bottom velocity $(0,0)$, top $(1,0)$, $p(0,0)=0$; $(u,v,p)=(y,0,0)$ | $u(1/2)=1/2$, flux $Q=1/2$, shear $\mu u_y=0.1$. Velocity outputs and pressure gauge. | [C], [P]; unit setup chosen here. |
| D2 — Poiseuille | $[0,4]\times[0,1]$, $\nu=0.1$, stationary no-slip plates, $p(0)=3.2$, $p(4)=0$, fully developed ends $u_x=v=0$; $(u,v,p)=(4y(1-y),0,0.8(4-x))$ | Centerline speed 1, mean speed/flux $2/3$, $p(2)=1.6$, wall shear magnitude 0.4. First Stokes, then restore convection in a separate check. | [P], [F], analytical formula only. |
| D3 — Kovasznay | $[-0.5,1]\times[-0.5,1.5]$, $\nu=1/Re$; exact velocity on all edges and one pressure anchor at $(1,-0.5)$; formula below | At $Re=40$: $(u,v,p)(0,0)=(0,0,0)$; $v(0,1/4)=-0.153384071467$; $u(1,0)=0.618536666468$, $p(1,0)=0.427242862585$. | [K]; rectangle, gauge, $Re$ sequence chosen here. |
| D4 — circular Couette | Annulus $1\le r\le2$, $\nu=0.1$; inner wall stationary, outer tangential speed 1, zero radial speed; $p(2)=0$; formula below | $u_\theta(1)=0$, $u_\theta(3/2)=5/9$, $u_\theta(2)=1$; $p(1)=-0.217202506169$. Curved no-slip walls. | [C]; radii, speeds, viscosity, gauge chosen here. |
| D5 — decaying shear | $x\in[0,2\pi]$ periodic, $y,t\in[0,1]$, no-slip at $y=0,1$, initial $(u,v)=(\sin\pi y,0)$, $p(0,0,t)=0$; $(u,v,p)=(e^{-0.1\pi^2t}\sin\pi y,0,0)$, $\nu=0.1$ | $u(y=1/2,t=1)=0.372707838853$. Add time while convection remains zero. | Derived shear solution; method [M]. |
| D6 — 2D Taylor vortex | Doubly periodic $[-\pi,\pi]^2$, $t\in[0,1]$, $\nu=0.01$; initial velocity from formula below, mean pressure zero at every time | At $(0,\pi/2,1)$: $u=-0.980198673307,v=0$; $p(0,0,1)=-0.480394719576$; mean kinetic energy 0.240197359788. | [T]; normalization chosen here. |

For Kovasznay,

$$
\lambda=Re/2-\sqrt{Re^2/4+4\pi^2},\quad
u=1-e^{\lambda x}\cos(2\pi y),\quad
v=\frac{\lambda}{2\pi}e^{\lambda x}\sin(2\pi y),\quad
p=\frac{1-e^{2\lambda x}}2.
$$

At $Re=40$, $\lambda=-0.963740544196$. Add separate $Re=10,20,40$
configurations, accepting each before increasing $Re$; the current runnable
case is fixed at 40. Poiseuille's vanishing convection cannot replace this gate.

For circular Couette, with the origin at the common cylinder center,

$$
u_r=0,\quad u_\theta=\frac23(r-r^{-1}),\quad
(u,v)=u_\theta(-\sin\theta,\cos\theta),\qquad
p(r)=\frac{2r^2-8\log r-2r^{-2}}9-\frac{8-8\log2-1/2}9.
$$

The radial balance is $p_r=u_\theta^2/r$. This is an annular rotating-wall
problem, not the DFG channel-cylinder problem. Passing it verifies curved-wall
machinery without claiming a solution for a different physical setup.

For the 2D Taylor vortex,

$$
u=-\cos x\sin y\,e^{-2\nu t},\quad
v=\sin x\cos y\,e^{-2\nu t},\quad
p=-\frac14(\cos2x+\cos2y)e^{-4\nu t},\qquad
\left\langle\frac{u^2+v^2}{2}\right\rangle=\frac14e^{-4\nu t}.
$$

Match periodic values and necessary derivatives; fix pressure's spatial mean
at each time. This exact 2D decay does not provide a reference for general 3D
Taylor–Green turbulence.

## What stays exploratory

The obstacle-free tunnel can separately check $\phi=x+C$, $u=1$, $v=0$.
Evaluate velocity derivatives rather than an arbitrary potential constant.
Audit the legacy pressure convention before using it for physical pressure.

The no-slip DFG cylinder and both bundled airfoil cases have no selected exact
solution for their full setups. Retain them for equation/geometry diagnostics;
do not label them accurate or promote them using published simulated forces.
Future experimental comparisons require a named measurement source, compatible
conditions, units, and uncertainty. Until then the accuracy status is
**unverified**. Residuals and mass conservation alone cannot close that gap.

Delay stiffness, long-time nonlinear oscillators, discontinuous shocks, shedding,
cavity benchmarks, turbulence, and 3D. A future case must first specify an
admissible reference; `solve_ivp`, DNS, and CFD exports cannot supply it.

## Sources

- **[L]** Lagaris, Likas & Fotiadis (1998),
  [Artificial Neural Networks for Solving Ordinary and Partial Differential Equations](https://arxiv.org/abs/physics/9705023),
  DOI [10.1109/72.712178](https://doi.org/10.1109/72.712178).
  A2 uses problem 2's exact solution; other elementary cases cite the method,
  not paper-reported parameter choices. Do not import the paper's FEM results.
- **[M]** Salari & Knupp (2000),
  [Code Verification by the Method of Manufactured Solutions](https://www.osti.gov/biblio/759450),
  SAND2000-1444, DOI [10.2172/759450](https://doi.org/10.2172/759450).
  Method reference; the project-defined cases above are our choices.
- **[B]** Cole (1951),
  [On a Quasi-linear Parabolic Equation Occurring in Aerodynamics](https://doi.org/10.1090/qam/42889),
  *Quarterly of Applied Mathematics* 9, 225–236. Analytical theory only.
- **[C]** Couette (1890),
  [Études sur le frottement des liquides](https://fr.wikisource.org/wiki/%C3%89tudes_sur_le_frottement_des_liquides),
  *Annales de chimie et de physique*, series 6, 21, 433–510.
  Historical primary text for viscous shear and concentric-cylinder flow.
- **[P]** MIT OpenCourseWare,
  [Couette & Poiseuille Flows](https://ocw.mit.edu/courses/2-25-advanced-fluid-mechanics-fall-2013/resources/mit2_25f13_couet_and_pois/),
  2.25 Advanced Fluid Mechanics (2013). Teaching notes, not a paper.
- **[F]** NETL MFiX,
  [FLD01: Steady, 2D Poiseuille flow, equation 3.2](https://mfix.netl.doe.gov/doc/vvuq-manual/main/html/fluid/fld-01.html).
  Analytical expression only, never its computed verification output.
- **[K]** Kovasznay (1948),
  [Laminar flow behind a two-dimensional grid](https://doi.org/10.1017/S0305004100023999),
  *Mathematical Proceedings of the Cambridge Philosophical Society* 44, 58–62;
  [primary paper PDF](https://www2.karlin.mff.cuni.cz/~prusv/vyuka/2013-2014/zs/downloads/kovasznay.pdf).
- **[T]** Taylor (1923),
  [On the decay of vortices in a viscous fluid](https://doi.org/10.1080/14786442308634295),
  *Philosophical Magazine*, series 6, 46, 671–674. Exact decaying vortices.

Bibliographic sources checked 2026-09-27. Each source's role is explicit:
analytical formula, verification method, or setup description. Publication
alone never makes simulation-generated comparison data admissible.

[L]: https://arxiv.org/abs/physics/9705023
[M]: https://www.osti.gov/biblio/759450
[B]: https://doi.org/10.1090/qam/42889
[C]: https://fr.wikisource.org/wiki/%C3%89tudes_sur_le_frottement_des_liquides
[P]: https://ocw.mit.edu/courses/2-25-advanced-fluid-mechanics-fall-2013/resources/mit2_25f13_couet_and_pois/
[F]: https://mfix.netl.doe.gov/doc/vvuq-manual/main/html/fluid/fld-01.html
[K]: https://doi.org/10.1017/S0305004100023999
[T]: https://doi.org/10.1080/14786442308634295

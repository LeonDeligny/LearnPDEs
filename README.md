# LearnPDEs — Physics-Informed Neural Networks in PyTorch

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1595a49bb6264b8fb196d55f1f544be0)](https://app.codacy.com/gh/LeonDeligny/LearnPDEs/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![License](https://img.shields.io/badge/license-GNU-blue.svg)](https://github.com/LeonDeligny/LearnPDEs/blob/main/LICENSE)

## **Project Goal**

The goal is to approximate solutions to ordinary and partial differential
equations using physics-informed neural networks (PINNs). Training uses only:

- **Physics Loss**: Derived from the governing equations (e.g., PDE = 0).
- **Boundary Conditions (BC) Loss**: Ensuring the solution satisfies the boundary constraints.
- **Initial Conditions (IC) Loss**: Prescribing the initial state for ODEs and time-dependent PDEs.

**Never use simulation data in any scenario, including for validation or
comparison.** No numerical ODE trajectories, CFD/FEM/FDM/spectral solution
fields, teacher-model predictions, simulation-derived boundary values, or
published simulation-derived scalar benchmarks are allowed. Geometry and
collocation coordinates are allowed; they are not solution labels.

Comparison values must come from analytical solutions or explicitly labelled
manufactured exact solutions. Exact interior values are evaluation-only and
must never enter training, pretraining, adaptive sampling, or early stopping.
Prescribed forcing and exact initial/boundary traces are legitimate problem
inputs. No experimental reference dataset is currently selected.

## Tutorials

Learn how to solve differential equations with physics-informed neural networks
in PyTorch. These worked examples explain the equations, automatic
differentiation, and loss functions, with runnable code and training animations.

- [Physics-informed neural networks for ODEs: a worked example](docs/tutorials/ode-pinn-pytorch.md): exponential growth and cosine oscillations.
- [Solve the Laplace equation with a PINN in PyTorch](docs/tutorials/laplace-pinn-pytorch.md): a two-dimensional boundary-value problem.
- [Implementing PINN boundary-condition losses in PyTorch](docs/tutorials/pinn-boundary-condition-losses-pytorch.md): value constraints, derivative constraints, and loss weighting.

Start with the [tutorial setup and running instructions](docs/tutorials/README.md).

## Run any scenario

All nine cases use one scenario catalog and a Typer CLI with grouped help,
validated parameters, and a saved configuration for each run:

```bash
uv sync --locked
uv run learnpdes --help
uv run learnpdes scenarios
uv run learnpdes examples
uv run learnpdes train --help
uv run learnpdes train laplace --epochs 5000 --points 21 --no-gif
```

Run a quick execution check of **every case** (including both bundled airfoil
formulations and the obstacle-free wind tunnel):

```bash
uv run learnpdes train all --epochs 1 --points 3 --resolution 5 --max-frames 2 --no-gif
```

Use `uv run learnpdes train all --dry-run` to inspect the resolved defaults before
training. `all` runs cases sequentially and stops on the first failure; completed
runs remain available. These short checks verify execution, not convergence.

`train --help` explains defaults, sample counts, and supported overrides, including
`--learning-rate`, `--hidden-dim`, `--hidden-layers`, `--seed`, `--threads`, and
output controls. Scenario-specific flags include `--cosinus-order`,
`--lbfgs-steps`, `--resample-every`, and `--mesh`. Use a single compatible case
when setting a nondefault derivative order, L-BFGS budget, or custom mesh.

`scenarios` displays the full catalog and defaults in a table; `examples` prints
copyable commands for every case and common parameter changes. For scripts,
`scenarios --json` and `train ... --dry-run` emit plain JSON. Typer also provides
shell completion through `--show-completion` and `--install-completion`.
Use `uv run learnpdes completion zsh` (or `bash`, `fish`, `powershell`, `pwsh`)
to print a script for an explicit shell, including from automated environments.

`uv run python -m learnpdes`, `python -m learnpdes.main`, and the existing
`examples.train_pinn` command accept the same arguments. Both batch export
commands now support the complete catalog and the same defaults;
`examples.generate_interactive` defaults to HTML-only output.
The original `learnpdes laplace ...`, `--scenario NAME`, and `--list-scenarios`
forms remain supported.

## Training results

Each training invocation creates a unique, portable run under
`assets/runs/<scenario>/<run-id>/`. It saves `training.gif`, `training.html`,
`run.json`, `loss.csv`, and `model.pt`; rendered PNGs stay in `frames/`.
Repeating a scenario preserves previous runs. The manifest records settings,
source revision, validation results when available, and export success/failure.
Numerical results and weights are saved before rendering begins.

```python
from learnpdes.main import main

path = main('laplace', epochs=1000)
print(path)  # assets/runs/laplace/<run-id>/training.html
```

GIFs show checkpoints from the untrained model through the last completed
optimizer update. HTML figures add hover, playback, reference/error views, and
fixed scales across training. Copy a run directory with its `plotly.min.js` to
share it offline. `--output-dir` changes the asset root. `--max-frames` controls
the checkpoint count; `--resolution` controls display samples independently of
training samples. `main(...)` uses `visualization_resolution` for this setting.

Kaleido is installed with the project. GIF export also needs Chrome and FFmpeg:

```bash
uv run plotly_get_chrome -y
# macOS: brew install ffmpeg; Ubuntu: sudo apt-get install ffmpeg
uv run python -m examples.train_pinn laplace --epochs 5000 --points 21
```

Use `--no-gif` or `main(..., save_gif=False)` for HTML-only output. The batch
runners `examples.generate_animations` and `examples.generate_interactive` use
the same run layout. `examples.compare_cosinus` groups comparisons under
`assets/comparisons/`. All working runs are ignored by Git.

## **Objectives**

The following scenarios are implemented. Quantitative accuracy must be checked
against an admissible exact reference independently of training loss.

1. **Simple ODEs** — implemented:
   - PINN: $f_{\theta}: \mathbb{R} \rightarrow \mathbb{R}$
   - ODE to approximate: $f' = f, f(0) = 1$
   - Physics Loss: $\lVert f_{\theta}' - f_{\theta}\rVert$
   - Boundary Loss: $\lVert f_{\theta}(0) - 1 \rVert$
   - Analytical solution: $\exp: \mathbb{R} \rightarrow \mathbb{R}$

   ![PINN training toward the exponential ODE solution](./assets/exponential.gif)

1. **Higher-Order ODEs** — implemented:
   - PINN: $f_{\theta}: \mathbb{R} \rightarrow \mathbb{R}$
   - ODE to approximate: $f'' = -f, f(0) = 1, f'(0) = 0$
   - Physics Loss: $\lVert f_{\theta}'' + f_{\theta} \rVert$
   - Boundary Loss: $\lVert f_{\theta}(0) - 1 \rVert, \lVert f'_{\theta}(0) \rVert$
   - Analytical solution: $\cos: \mathbb{R} \rightarrow \mathbb{R}$

   ![PINN training toward the cosine ODE solution](./assets/cosinus.gif)

1. **Laplace Equation** — implemented:
   - PINN: $f_{\theta}: [0, 1]^2 \rightarrow \mathbb{R}$
   - PDE to approximate: $\Delta f = 0$
   - Dirichlet boundary conditions: $f(\cdot, 0) = 0, f(\cdot, 1) = \sin(\pi x), f(0, \cdot) = 0, f(1, \cdot) = 0$
   - Physics loss: $\lVert \Delta f_{\theta} \rVert$
   - Boundary loss: $\lVert f_{\theta}(\cdot, 0) \rVert, \lVert f_{\theta}(\cdot, 1) - \sin(\pi x) \rVert, \lVert f_{\theta}(0, \cdot) \rVert, \lVert f_{\theta}(1, \cdot) \rVert$
   - Analytical solution: $f(x, y) = \sin(\pi x) \sinh(\pi y)/\sinh(\pi)$

   ![PINN training toward the Laplace equation solution on a unit square](./assets/laplace.gif)

1. **Potential, irrotational flow** — exploratory:
   - Wind tunnel scenario with no geometry
   - PINN: $\phi_{\theta}: [0, 4] \times [0, 1] \rightarrow \mathbb{R}$
   - PDE to approximate: $\Delta\phi=0$ (incompressibility).
   - Dirichlet boundary conditions: $(u, v)(\partial([0, 4] \times [0, 1])) = (1, 0)$
   - No simulation data. Compare velocity only with the analytical uniform
     field $(1,0)$; independent accuracy checks and a pressure-convention audit
     remain pending. Both bundled airfoil formulations are also exploratory;
     their meshes provide geometry, not admissible solution references.

   ![Training Process](./assets/wind_tunnel_no_geometry.gif)

1. **Navier–Stokes: Poiseuille flow between flat plates** — implemented:

   Use scenario **`poiseuille`** for steady, incompressible viscous flow in
   $[0,4]\times[0,1]$. The network predicts $(u,v,p)$ directly. In dimensionless
   model units, density is $\rho=1$, dynamic viscosity is $\mu=0.1$, and pressure
   drops from $3.2$ at the inlet to $0$ at the outlet. The residuals are

   $$
   u_x+v_y=0,\qquad
   \rho(u u_x+v u_y)+p_x-\mu(u_{xx}+u_{yy})=0,\qquad
   \rho(u v_x+v v_y)+p_y-\mu(v_{xx}+v_{yy})=0.
   $$

   Both plates ($y=0$ and $y=1$) enforce **no slip and no penetration**, $u=v=0$.
   At the inlet and outlet, $u_x=0$ and $v=0$ specify fully developed flow;
   the prescribed pressures supply the forcing and fix the pressure reference.
   There is no body force and no prescribed parabolic velocity target in training.

   For $G=-dp/dx=\Delta p/L$, the exact solution is

   $$
   u(y)=\frac{G}{2\mu}y(H-y)=4y(1-y),\qquad
   v=0,\qquad p(x)=0.8(4-x).
   $$

   It has centerline speed $u(1/2)=1$ and mean speed $2/3$. This pressure–viscosity
   balance and parabolic profile follow the
   [MFiX plane Poiseuille verification case](https://mfix.netl.doe.gov/doc/vvuq-manual/main/html/fluid/fld-01.html),
   with the model constants and fully developed end conditions specified above.

   Train on a 21 × 21 collocation grid and evaluate against the exact solution on
   a separate 41 × 41 grid:

   ```bash
   uv run --group export python -m examples.train_pinn poiseuille --epochs 5000 --points 21
   ```

   The runner saves GIF and HTML under `assets/runs/poiseuille/<run-id>/`, prints their
   paths, and reports aggregate errors and individual velocity/pressure errors.
   Use `--output-dir` to choose another directory and `--max-frames` to control the
   number of checkpoints (40 by default). PNGs remain in the run’s `frames/` directory.
   For $v$, whose exact value is zero everywhere, it reports absolute errors only.
   Use `--epochs 10` for a quick execution check. The optimized objective is
   `3 * physics_loss + boundary_loss`; the exact solution is used only for evaluation.

   To export an interactive figure with prediction, reference, signed-error views,
   and training loss:

   ```python
   from learnpdes.main import main

   path = main('poiseuille', epochs=5000, output_dir='assets')
   print(path)  # assets/runs/poiseuille/<run-id>/training.html
   ```

   The CLI and Python entry point both default to 21 × 21 training points and an
   independent 51 × 51 display grid. Parameters and the reference solution are in
   [learnpdes/poiseuille.py](learnpdes/poiseuille.py); the
   [benchmark tests](tests/test_poiseuille.py) verify the exact solution, pressure
   forcing, no-slip walls, automatic derivatives, and training/plot integration.

1. **Navier–Stokes: Kovasznay flow on a rectangle** — implemented:

   Use scenario **`kovasznay`** for coupled nonlinear, steady, incompressible 2D
   Navier–Stokes on $[-0.5,1]\times[-0.5,1.5]$, with **no obstacle**. The network
   predicts $(u,v,p)$ directly. The dimensionless setup uses $Re=40$, unit density,
   kinematic viscosity $\nu=1/Re=0.025$, and zero body force:

   $$
   u u_x+v u_y+p_x-\nu(u_{xx}+u_{yy})=0,
   \qquad
   u v_x+v v_y+p_y-\nu(v_{xx}+v_{yy})=0,
   \qquad
   u_x+v_y=0.
   $$

   The exact solution is

   $$
   \lambda=\frac{Re}{2}-\sqrt{\frac{Re^2}{4}+4\pi^2},\qquad
   u=1-e^{\lambda x}\cos(2\pi y),\qquad
   v=\frac{\lambda}{2\pi}e^{\lambda x}\sin(2\pi y),\qquad
   p=\frac{1-e^{2\lambda x}}{2}.
   $$

   Both velocity components take their exact values on all four rectangle edges;
   these edges are not stationary no-slip walls. The exact pressure at the
   bottom-right corner $(1,-0.5)$ fixes the additive pressure constant. The loss
   sums the three mean squared PDE residuals and ten times each boundary and
   pressure penalty. Interior exact values are reserved for evaluation.

   Train with 961 independent interior samples and 124 samples per edge. Adam
   resamples every 100 steps; L-BFGS holds its samples fixed. The 4 × 64 tanh MLP
   reports aggregate and per-component errors on 2,048 unseen uniform points:

   ```bash
   uv run python -m examples.train_pinn kovasznay --epochs 1500 --lbfgs-steps 1500 --points 31
   ```

   The command saves GIF, HTML, metadata, and weights under
   `assets/runs/kovasznay/<run-id>/`. PNG checkpoints remain in its `frames/` folder. Use `--max-frames 40` and
   `--resolution 51` to control checkpoint count and display resolution, or
   `--no-gif` for HTML only. For a quick setup check, use `--epochs 10`; this
   verifies execution, not convergence. The Python entry point also saves a GIF
   and an interactive figure with velocity, pressure, reference fields, signed
   errors, and training loss:

   ```python
   from learnpdes.main import main

   path = main('kovasznay', epochs=1000, output_dir='assets')
   print(path)  # assets/runs/kovasznay/<run-id>/training.html; GIF beside it
   ```

   The CLI and Python entry point both default to 961 interior samples, 124
   samples per boundary, and an independent 51 × 51 display grid. Benchmark constants and the NumPy/Torch
   exact solution live in [learnpdes/kovasznay.py](learnpdes/kovasznay.py).
   The [benchmark tests](tests/test_kovasznay.py) check exact residual cancellation,
   both momentum equations, all velocity boundaries, the pressure gauge, and
   training/evaluation integration. The equations and rectangle follow the
   [DeepXDE Kovasznay implementation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Kovasznay.flow.html#implementation),
   using $Re=40$ and a single pressure reference point here. The analytical
   solution originates in [Kovasznay (1948)](https://doi.org/10.1017/S0305004100023999).
   No simulation data are used; compare with the exact formula only.

1. **Navier–Stokes: low-Reynolds-number flow around a cylinder** — exploratory

   Scenario **`cylinder`** implements the steady DFG 2D-1 channel benchmark at
   $Re=20$. It uses direct $(u,v,p)$ outputs, a 4 × 64 tanh MLP, exact inlet and
   no-slip wall enforcement, and a do-nothing outlet that fixes pressure.

   ```bash
   uv run python -m examples.train_pinn cylinder --epochs 3000 --lbfgs-steps 3000 --points 45 --no-gif
   ```

   Fluid runs also save each PDE
   and boundary loss in `residuals.csv`. The evaluator reports predicted drag,
   lift, pressure drop, and mass balance as diagnostics. Published DFG reference
   values come from numerical simulations and are excluded from comparison.
   No simulation data are allowed, even for validation. With no selected exact
   reference for this setup, cylinder solution accuracy remains unverified.

### **Installation**

Ensure you have the following installed on your system:

- ✔️ Python 3.12
- ✔️ CPU, or MPS support for acceleration on macOS
- ✔️ [uv](https://docs.astral.sh/uv/)
- ✔️ Additional Python dependencies

1. **Clone the Repository**:

   Clone this repository to your local machine:
   ```bash
   git clone https://github.com/LeonDeligny/LearnPDEs.git
   cd LearnPDEs
   ```

2. **Install uv**:

   Follow the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

3. **Install dependencies**:

   Install the dependencies with:
   ```bash
   uv sync
   ```

   This creates a `.venv` virtual environment and installs the dependencies from `uv.lock`.

4. **Run the tests**:

   ```bash
   uv run python -m unittest discover -s tests -p "test*.py"
   ```

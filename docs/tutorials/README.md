---
title: Install and run the LearnPDEs PINN tutorials
short_title: Getting started
description: Set up LearnPDEs with Python and uv, run PyTorch PINN examples on a CPU or Mac, and check ODE, Laplace, and Poiseuille flow solutions against exact references.
---

# Getting started

**Never use simulation data in any scenario**, including comparison values
from published numerical simulations. Training uses equations and prescribed
initial/boundary conditions only; analytical interior values are evaluation-only.
The [validation plan](../validation.md) records each of the registered scenarios'
allowed references, papers, exact comparison values, and a gradual progression.
Cylinder and airfoil runs are exploratory; no admissible solution reference is
selected for their full setups. Their diagnostics do not establish accuracy.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). CPU or Mac MPS.

```bash
git clone https://github.com/LeonDeligny/LearnPDEs.git
cd LearnPDEs
uv sync --locked
```

Training saves a GIF, interactive HTML, settings, loss history, and weights under
`assets/runs/<scenario>/<run-id>/`. Each invocation preserves previous runs.
For GIF export, install Chrome with `uv run plotly_get_chrome -y` and FFmpeg
with `brew install ffmpeg` (macOS) or `sudo apt-get install ffmpeg` (Ubuntu).
Add `--no-gif` to a training command for HTML-only output without those tools.
Use `--output-dir` to choose another folder and `--max-frames` to set the maximum
number of animation checkpoints; the initial and final states are included.

Every new training figure shows its actual loss-evaluation coordinates on the
first prediction panel. Navy marks are PDE points, orange marks are boundary
points, red stars are initial/pressure anchors, and purple crosses are flux
quadrature points. For ODEs, marks along the bottom show x locations, not predicted
values. HTML provides **Show points / Hide points** buttons; GIFs show the points
throughout. Adam resampling updates the displayed locations at each saved
checkpoint; L-BFGS locations stay fixed. The field display uses a separate grid.

`collocation.json` stores these checkpoint coordinates, deduplicating unchanged
sets. Counts describe loss roles: a boundary node may also evaluate the PDE.
Older runs without this recording cannot recover their exact training locations;
their standalone plots explicitly report that the points were not recorded.

## Run any scenario

```bash
uv run learnpdes --help
uv run learnpdes scenarios
uv run learnpdes examples
uv run learnpdes train --help
uv run learnpdes train laplace --no-gif
```

The CLI uses Typer for typed parameters, validation, help panels, and shell
completion, with Rich tables for the scenario catalog. Run `learnpdes` without
arguments to see the command overview.

| Command | Purpose |
| --- | --- |
| `learnpdes --help` | Command overview and completion options |
| `learnpdes scenarios` | All cases, descriptions, and training defaults |
| `learnpdes scenarios --json` | Machine-readable scenario catalog |
| `learnpdes examples` | Copyable commands for every case and common overrides |
| `learnpdes train --help` | Parameters grouped by training, network, reproducibility, scenario, and output |
| `learnpdes train NAME` | Train one case, or every case with `all` |
| `learnpdes train NAME --dry-run` | Resolved configuration as JSON, without training |
| `learnpdes completion zsh` | Print completion for a chosen shell, without detection or installation |

Prefix these commands with `uv run` when using the project environment. `-h`
also displays help. To inspect completion for the current shell, use
`uv run learnpdes --show-completion`. To install it, run
`uv run learnpdes --install-completion` and restart the shell; this updates the
shell's completion configuration. Completion suggests commands, options, and
scenario names with descriptions. Use the explicit `train` form for completion.
In automated environments where shell detection is unavailable, use
`uv run learnpdes completion zsh`; supported names are `bash`, `zsh`, `fish`,
`powershell`, and `pwsh`. This only prints the script and does not change shell files.

Redirected JSON output contains no tables, color escapes, or progress messages:

```bash
uv run learnpdes scenarios --json > scenarios.json
uv run learnpdes train all --dry-run > configurations.json
```

Each case owns its setup under `learnpdes/scenarios/` and is registered in
`learnpdes/scenarios/registry.py`. The CLI,
Python API, and example runners use the same model construction, training,
evaluation, defaults, and export pipeline.

| Scenario | Problem | Adam updates | Training points (`--points`) |
| --- | --- | ---: | ---: |
| `exponential` | Exponential growth ODE | 10000 | 256 |
| `forced-linear` | Forced linear ODE (Lagaris problem 2) | 5000 | 64 |
| `logistic` | Nonlinear logistic growth ODE | 5000 | 64 |
| `cosinus` | Harmonic oscillator | 5000 | 64 |
| `laplace` | Laplace equation on a unit square | 5000 | 21 |
| `poiseuille` | Pressure-driven viscous channel flow | 5000 | 21 |
| `kovasznay` | Exact Navier–Stokes benchmark, Re=40 | 10000 | 31 |
| `circular-couette` | Exact annular viscous flow, Re=10 | 1500 | 32 |
| `cylinder` | DFG channel/cylinder benchmark, Re=20 | 5000 | 45 |
| `wind_tunnel_no_geometry` | Potential flow without an obstacle | 1000 | 32 |
| `potential-flow` | Potential flow around an airfoil | 5000 | 32 |
| `solenoidal-flow` | Streamfunction flow around an airfoil | 5000 | 32 |

Inspect the resolved configuration without training or creating output files:

```bash
uv run learnpdes train all --dry-run
```

Run every case with a small budget to check installation and exports:

```bash
uv run learnpdes train all --epochs 1 --points 3 --resolution 5 --max-frames 2 --no-gif
```

`all` runs sequentially and stops on the first failure, retaining earlier runs.
It resets the seed for each case. These budgets check execution; use the tutorial
budgets and numerical error measurements to assess convergence.

### Change training parameters

```bash
uv run learnpdes train laplace --epochs 2000 --points 31 --learning-rate 0.0005 --hidden-dim 32 --hidden-layers 3 --seed 7 --no-gif
uv run learnpdes train cosinus --cosinus-order 6 --epochs 5000 --no-gif
uv run learnpdes train kovasznay --epochs 1500 --lbfgs-steps 1500 --resample-every 100 --no-gif
```

- `--points`: samples for ODEs (`exponential` and `cosinus` also add the initial
  point), points per axis
  for rectangular grids, or the square root of the interior sample count for
  cylinder/Kovasznay/circular-Couette (each boundary has `4 * points` samples). Airfoil cases keep
  the SU2 vertices and add a `points × points` grid, so a small value still uses
  the complete mesh.
- `--learning-rate`, `--hidden-dim`, `--hidden-layers`: Adam step size and Tanh
  network width/depth. Defaults are `0.001`, 20 units (64 for cylinder/Kovasznay/circular-Couette),
  and four hidden layers.
- `--seed`, `--threads`: random seed (default `0`) and Torch CPU threads (default
  `1`). Repeating a seed on the same environment reproduces initialization and
  sampling; numerical results can differ across hardware and Torch versions.
- `--cosinus-order`: an even integer at least 2, for `cosinus` only.
- `--lbfgs-steps`: nonnegative updates after Adam, for cylinder/Kovasznay/circular-Couette only.
  `--resample-every` controls their Adam sampling interval. L-BFGS uses fixed samples.
- `--mesh`: an alternate SU2 airfoil mesh for `potential-flow` or
  `solenoidal-flow`. It must have fluid triangles/quads and `inlet`, `outlet`,
  `wall`, and `airfoil` markers. The bundled mesh is installed with the package,
  so the defaults work outside the repository too.
- `--output-dir`, `--max-frames`, `--resolution`: asset root, saved checkpoint
  limit, and independent display sampling. Airfoil visualization follows mesh
  connectivity; its sample count is not controlled by `--resolution`.
- `--no-gif`: save interactive HTML without Chrome or FFmpeg. GIF is enabled by
  default, and its prerequisites are checked before training.

Invalid values and incompatible scenario-specific flags fail before any run
starts. With `all`, common overrides apply to every case; select a single case
to change its derivative order, L-BFGS budget, or mesh.

Every `run.json` records the resolved configuration under `settings.config`,
alongside the model, loss history, source revision, and evaluation results.
Run directories are unique, so rerunning a command preserves previous results.

### Python API

```python
from learnpdes.config import RunConfig
from learnpdes.training import train

trainer = train(
    RunConfig(
        'laplace',
        epochs=2000,
        points=31,
        learning_rate=0.0005,
        resolution=51,
        save_gif=False,
    )
)
html = trainer.html_path
```

The Python API uses `RunConfig.points` for CLI `--points` and
`RunConfig.resolution` for `--resolution`. Python names
`'potential flow'` and `'solenoidal flow'` are accepted as aliases.
Unspecified epochs and point counts now use the scenario table above.

`uv run python -m learnpdes` accepts the same arguments as `uv run learnpdes`.
Use `train --help` for the complete parameter reference. The original
scenario-first commands (`learnpdes laplace ...`), `--scenario NAME`, and
`--list-scenarios` remain supported.

### Work with saved runs

All workflows use the `learnpdes` CLI. Run any command with `--help` for its
options:

```bash
uv run learnpdes train all                 # HTML and GIFs for every scenario
uv run learnpdes train all --no-gif        # HTML only
uv run learnpdes refine path/to/fluid-run --points 81 --lbfgs-steps 3000
uv run learnpdes verify path/to/fluid-run
uv run learnpdes plot path/to/cylinder-run --png
uv run learnpdes compare-cosinus --orders 2 --orders 4 --seeds 0
uv run learnpdes publish path/to/completed-run
```

`refine` and `verify` support fluid runs; `plot` currently exports cylinder
diagnostics. `publish` copies a selected run into `assets/examples/` for the
documentation; replacing an existing example requires `--replace`.

Problem definitions live in `learnpdes/scenarios/`, workflow commands in
`learnpdes/commands/`, and selected documentation outputs in `assets/examples/`.

## Choose an equation

### First-order ODE progression (A1–A3)

All three cases are runnable. They train on the equation and the prescribed
initial value, with loss $3\,\mathrm{mean}(r^2)+(f_\theta(0)-f(0))^2$.
Exact solution values are used only for evaluation and plots.

| Step / scenario | Equation and initial value | Domain | Exact solution | $f(1)$ |
| --- | --- | --- | --- | --- |
| A1 / `exponential` | $f'=f$, $f(0)=1$ | $[-3,3]$ | $e^x$ | 2.718281828459 |
| A2 / `forced-linear` | $f'+f/5=e^{-t/5}\cos t$, $f(0)=0$ | $[0,2]$ | $e^{-t/5}\sin t$ | 0.688938173085 |
| A3 / `logistic` | $f'=f(1-f)$, $f(0)=1/2$ | $[0,2]$ | $1/(1+e^{-t})$ | 0.731058578630 |

A2 follows [Lagaris, Likas & Fotiadis, section 4.1.2, problem 2](https://arxiv.org/abs/physics/9705023).
The sample counts and Adam budgets in this project are our defaults, not a
reproduction of that paper's training setup. A3 uses unit growth rate and
carrying capacity.

```bash
uv run learnpdes train exponential --no-gif
uv run learnpdes train forced-linear --no-gif
uv run learnpdes train logistic --no-gif
```

Each run reports RMSE, maximum error, and relative L2 error against its exact
formula on a separate 201-point grid. This is a basic comparison; the full
[multi-seed acceptance gate](../validation.md#acceptance-gate-for-every-step)
remains to be completed.

### Worked tutorials

| Tutorial | Equation |
| --- | --- |
| [Exponential growth](exponential-ode-pinn-pytorch.md) | $f'=f$ |
| [Harmonic oscillator](harmonic-oscillator-pinn-pytorch.md) | $f''+f=0$ |
| [Laplace equation](laplace-pinn-pytorch.md) | $\nabla^2 u=0$ |
| [Poiseuille flow](poiseuille-flow-pinn-pytorch.md) | Steady incompressible Navier–Stokes; $\mu u_{yy}=p_x$ for the exact channel flow |
| [Circular Couette](circular-couette-pinn-pytorch.md) | Exact annular viscous flow with curved no-slip walls |
| [Cylinder flow](cylinder-flow-pinn-pytorch.md) | Steady incompressible Navier–Stokes around a no-slip cylinder at Re=20 |

Each tutorial gives the conditions, training command and PyTorch loss.
Use `--epochs 10` for an installation check, not an accuracy test.

## Read the results

For cases with an exact reference, compare RMSE, maximum error and relative L2
error with that formula. Never substitute simulation data for a missing reference.
Training loss measures equation fit, not solution accuracy.

The scalar and Poiseuille examples use four hidden layers of 20 `Tanh` units and Adam
(learning rate `0.001`, seed `0`). Evaluation grids include points between
training samples; some coordinates overlap.

Kovasznay, circular Couette, and cylinder use four layers of 64 `Tanh` units, independent random
interior/boundary sampling, and optional L-BFGS after Adam. Their evaluator uses
fresh points and reports each PDE residual separately.

[Training source](../../learnpdes/training.py) · [Network](../../learnpdes/model/pinn.py)

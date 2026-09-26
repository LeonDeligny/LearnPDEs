# LearnPDEs tutorials: physics-informed neural networks in PyTorch

Learn to approximate ordinary and partial differential equations with a
physics-informed neural network (PINN). These tutorials use LearnPDEs' existing
PyTorch model and loss functions, explain the mathematics, and compare the
predictions with known analytical solutions.

## Choose a tutorial

| Tutorial | What you will learn |
| --- | --- |
| [Physics-informed neural networks for ODEs: a worked example](ode-pinn-pytorch.md) | Solve exponential growth and cosine oscillations using first and second derivatives. |
| [Solve the Laplace equation with a PINN in PyTorch](laplace-pinn-pytorch.md) | Construct a two-dimensional PDE residual and enforce four Dirichlet boundaries. |
| [Implementing PINN boundary-condition losses in PyTorch](pinn-boundary-condition-losses-pytorch.md) | Implement value and derivative constraints, inspect their losses, and combine them with the physics residual. |

## Install and run

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
LearnPDEs selects MPS on supported Macs and otherwise uses the CPU; a GPU is not
required. From a terminal:

```bash
git clone https://github.com/LeonDeligny/LearnPDEs.git
cd LearnPDEs
uv sync --locked
uv run python -m examples.train_pinn exponential --epochs 10000 --points 256
uv run python -m examples.train_pinn cosinus --epochs 5000 --points 64
uv run python -m examples.train_pinn laplace --epochs 5000 --points 21
```

If you already have a checkout, run the last four commands from its root. The
scenario identifier `cosinus` is the name used by LearnPDEs for the cosine ODE.
For a quick installation check, use `--epochs 10` in any of the commands;
ten updates check that training runs, but do not establish solution accuracy.

The [tutorial runner](../../examples/train_pinn.py) uses the same `PINN`, `Loss`,
and `load_scenario` implementations as the main application. It trains without
opening plots or writing images. Each page embeds the repository's existing
training animations; they illustrate earlier runs and are not promised outputs
of these commands.

## Read the results

Each run prints the initial root mean squared error (RMSE), training losses,
final RMSE, and maximum absolute error. The errors compare predictions with the
analytical solution on a separate evaluation grid: 201 points for an ODE and
41 by 41 points for Laplace. Some evaluation coordinates also occur in the
training grid. The exact solution is never supplied to the optimizer as a
training target.

The optimized objective is `3 * physics_loss + boundary_loss`, matching
[`Loss.process`](../../learnpdes/model/loss.py). It is different from prediction
RMSE. Inspect both: a small residual on the sampled coordinates alone does not
guarantee accuracy throughout the domain.

Training uses Adam with a learning rate of `0.001` and seed `0`. Try `--seed 1`
to compare another initialization, or increase `--epochs` and `--points` to
investigate convergence. Results and runtime depend on the device and PyTorch
version. For Laplace, doubling `--points` roughly quadruples the number of
collocation points.

## How the examples fit the repository

- [`loadscenarios.py`](../../learnpdes/utils/loadscenarios.py) supplies coordinates,
  boundary masks, and analytical reference functions.
- [`pinn.py`](../../learnpdes/model/pinn.py) defines the fully connected network,
  using four hidden layers of 20 `Tanh` units in these examples.
- [`loss.py`](../../learnpdes/model/loss.py) computes automatic derivatives and
  the equation-specific losses.
- [`train_pinn.py`](../../examples/train_pinn.py) provides the tutorial commands
  and a small evaluation loop.

[Back to the LearnPDEs README](../../README.md)

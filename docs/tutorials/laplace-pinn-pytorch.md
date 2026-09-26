# Solve the Laplace equation with a PINN in PyTorch

This tutorial solves the two-dimensional Laplace equation on a unit square
using a physics-informed neural network (PINN) in PyTorch. You will construct
the PDE residual with automatic differentiation, enforce Dirichlet boundary
conditions, and compare the trained LearnPDEs model with an analytical solution.

[All tutorials and setup](README.md) · [Runnable source](../../examples/train_pinn.py)

## Run the example

Follow the [installation instructions](README.md#install-and-run), then run
from the repository root:

```bash
uv run python -m examples.train_pinn laplace --epochs 5000 --points 21
```

This creates a 21 by 21 grid of 441 collocation points, including the square's
edges. The network maps two coordinates `(x, y)` to one scalar value `u` using
four hidden layers of 20 `Tanh` units. A CPU is sufficient; LearnPDEs uses MPS
automatically when it is available.

## Specify the PDE and boundary conditions

We seek a function $u(x,y)$ on $[0,1]^2$ satisfying

$$
\Delta u = u_{xx}+u_{yy}=0.
$$

The four boundary conditions are

$$
u(x,0)=0,\qquad u(x,1)=\sin(\pi x),\qquad
u(0,y)=0,\qquad u(1,y)=0.
$$

This problem has the analytical solution

$$
u(x,y)=\frac{\sin(\pi x)\sinh(\pi y)}{\sinh(\pi)}.
$$

It satisfies the boundary values, and its second derivatives cancel:
$u_{xx}=-\pi^2u$ while $u_{yy}=\pi^2u$. We use this expression to evaluate
the trained network. The optimizer receives the PDE and boundary conditions,
not samples of this analytical solution inside the domain.

![LearnPDEs PINN training toward the analytical Laplace equation solution on the unit square](../../assets/laplace.gif)

The existing animation shows an earlier training run. The tutorial command
prints numerical metrics and does not generate image files.

## Form the physics and boundary losses

Let the network prediction be $u_\theta(x,y)$. The physics loss is the mean
squared Laplacian over the collocation grid:

$$
\mathcal L_{\mathrm{physics}}=
\frac{1}{N}\sum_{i=1}^{N}
\left(u_{\theta,xx}(x_i,y_i)+u_{\theta,yy}(x_i,y_i)\right)^2.
$$

The boundary loss is the sum of four mean squared errors, one for each edge:

$$
\mathcal L_{\mathrm{boundary}}=
\operatorname{MSE}(u_\theta(x,0),0)
+\operatorname{MSE}(u_\theta(x,1),\sin(\pi x))
+\operatorname{MSE}(u_\theta(0,y),0)
+\operatorname{MSE}(u_\theta(1,y),0).
$$

Using a separate mean per edge makes each term an average over that edge's
samples. The corner points appear on both adjoining edges; the specified
targets agree at every corner.

The loss implementation calls the left edge `inlet` and the right edge
`outlet`. These are shared mask names, not extra flow conditions for this PDE.

## Inspect a complete PyTorch loss calculation

Save this as `inspect_laplace.py` in the repository root, then run
`uv run python inspect_laplace.py`. It builds the problem, computes both
second derivatives, and checks that the explicit loss agrees with LearnPDEs.

```python
import torch

from examples.train_pinn import build_problem

torch.manual_seed(0)
model, problem, _ = build_problem('laplace', points=21)
u = model(problem.inputs)

du_dx = problem.partial_derivative(u, problem.x)
du_dy = problem.partial_derivative(u, problem.y)
d2u_dx2 = problem.partial_derivative(du_dx, problem.x)
d2u_dy2 = problem.partial_derivative(du_dy, problem.y)
physics_loss = (d2u_dx2 + d2u_dy2).square().mean()

top_target = torch.sin(torch.pi * problem.x[problem.top_mask])
boundary_loss = (
    u[problem.bottom_mask].square().mean()
    + (u[problem.top_mask] - top_target).square().mean()
    + u[problem.inlet_mask].square().mean()
    + u[problem.outlet_mask].square().mean()
)

total_loss = 3 * physics_loss + boundary_loss
reference_loss, _, _, _ = problem.laplace_loss()
torch.testing.assert_close(total_loss, reference_loss)
print('Physics loss:', physics_loss.item())
print('Boundary loss:', boundary_loss.item())
print('Total loss:', total_loss.item())
```

The repository writes the physics term as `MSE(u_xx, -u_yy)`, which is
algebraically the same as `mean((u_xx + u_yy)**2)`.

Coordinates require gradients, and `partial_derivative` uses
`create_graph=True` so that derivatives remain differentiable. Keep gradient
tracking enabled while computing the PDE loss. `torch.no_grad()` is suitable
for comparing predicted values with the exact solution, but not for computing
this residual. [PyTorch gradient modes](https://docs.pytorch.org/docs/stable/notes/autograd.html#grad-modes)

## Evaluate convergence

The training command prints initial and final RMSE against the analytical
solution on a separate 41 by 41 grid, plus maximum absolute error. This grid
includes points between training coordinates as well as some shared points.
The maximum error helps reveal discrepancies that an average can hide.

Increasing `--points` improves the spatial sampling of the residual, but also
increases cost: `--points 41` uses 1,681 training coordinates. Increasing
`--epochs` gives the optimizer more updates. Neither change guarantees a
particular error, so compare the printed metrics between runs.

If the model approaches zero everywhere, inspect the top-edge loss: zero has
a perfect Laplace residual but fails the nonzero top boundary. If the boundary
fit looks good while the interior is inaccurate, inspect the physics loss and
evaluate on a denser grid. These checks separate different causes of a poor
approximation.

Continue with [implementing boundary-condition losses](pinn-boundary-condition-losses-pytorch.md),
or start with [the one-dimensional ODE examples](ode-pinn-pytorch.md).

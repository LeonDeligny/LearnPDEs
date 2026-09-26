# Implementing PINN boundary-condition losses in PyTorch

Boundary and initial conditions select the solution a physics-informed neural
network (PINN) should learn. This tutorial shows how to implement value and
derivative constraints in PyTorch, inspect each loss term, and combine them
with a PDE or ODE residual using LearnPDEs.

[All tutorials and setup](README.md) · [Loss implementation](../../learnpdes/model/loss.py)

## Why the physics loss is not enough

For the [Laplace example](laplace-pinn-pytorch.md), the function $u=0$ has zero
physics loss everywhere, but fails the top boundary $u(x,1)=\sin(\pi x)$.
For the [cosine ODE](ode-pinn-pytorch.md), all functions
$A\cos(x)+B\sin(x)$ satisfy $f''+f=0$. Constraints determine $A$ and $B$.

LearnPDEs enforces constraints through loss penalties. This is often called
**soft enforcement**: training encourages the constraints to hold, but finite
optimization error means they may not hold exactly.

## Dirichlet conditions: constrain the predicted value

A Dirichlet condition prescribes $u=g$ on a boundary. With sampled boundary
coordinates $z_j$, its loss is

$$
\mathcal L_D=\frac{1}{M}\sum_{j=1}^{M}
\left(u_\theta(z_j)-g(z_j)\right)^2.
$$

After [installing the repository](README.md#install-and-run), save this as
`inspect_dirichlet.py` in the repository root and run
`uv run python inspect_dirichlet.py`:

```python
import torch

from examples.train_pinn import build_problem

torch.manual_seed(0)
model, problem, _ = build_problem('laplace', points=21)
u = model(problem.inputs)

# Every prediction and target has shape (number_of_edge_points, 1).
top_values = u[problem.top_mask]
top_target = torch.sin(torch.pi * problem.x[problem.top_mask])
edge_losses = {
    'bottom': u[problem.bottom_mask].square().mean(),
    'top': (top_values - top_target).square().mean(),
    'left': u[problem.inlet_mask].square().mean(),
    'right': u[problem.outlet_mask].square().mean(),
}
boundary_loss = sum(edge_losses.values())

for edge, loss in edge_losses.items():
    print(f'{edge}: {loss.item():.6e}')
print('Total boundary loss:', boundary_loss.item())
```

Using separate means preserves an explicit weight for each edge even if the
numbers of points differ. Averaging all boundary samples together instead
weights edges according to their sample counts. Neither choice eliminates the
need to check the fit along each edge.

The loader uses exact endpoint comparisons such as `y == 1` because its regular
grid contains those endpoints explicitly. For noisy or imported coordinates,
use reliable boundary labels or a tolerance-based selection. An empty boundary
mask produces an undefined mean rather than a useful constraint.

## Derivative conditions: constrain a slope or normal derivative

For the cosine initial-value problem, LearnPDEs imposes $f(0)=1$ and $f'(0)=0$.
The point `x = 0` lies inside the sampled interval `[-3, 3]`; these are initial
conditions, not geometric boundary conditions at the interval's endpoints.
The derivative calculation nevertheless demonstrates the mechanism used for
derivative boundary losses.

Save this independent example as `inspect_derivative.py` and run
`uv run python inspect_derivative.py`:

```python
import torch

from examples.train_pinn import build_problem

torch.manual_seed(0)
model, problem, _ = build_problem('cosinus', points=64)
f = model(problem.inputs)
df_dx = problem.partial_derivative(f, problem.x)
d2f_dx2 = problem.partial_derivative(df_dx, problem.x)

value_loss = (f[problem.zero_mask] - 1).square().mean()
slope_loss = df_dx[problem.zero_mask].square().mean()
physics_loss = (d2f_dx2 + f).square().mean()
boundary_loss = value_loss + slope_loss
total_loss = 3 * physics_loss + boundary_loss

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad(set_to_none=True)
total_loss.backward(retain_graph=True)
optimizer.step()
print('Value loss before the update:', value_loss.item())
print('Slope loss before the update:', slope_loss.item())
```

The shared `Loss` object caches coordinate graphs and, for Laplace, a boundary
target graph. This example and the tutorial runner follow the existing
`Trainer.train` by retaining the graph during backward. `retain_graph=True`
is specific to this reuse; it is not a general requirement for every PINN
training loop. `create_graph=True` in the derivative helper serves a different
purpose: it enables differentiation through the derivatives.
[PyTorch autograd reference](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)

For a Neumann boundary condition in two dimensions, prescribe the outward
normal derivative instead of a single coordinate derivative:

$$
\nabla u\cdot n = u_x n_x + u_y n_y = h,
\qquad
\mathcal L_N = \operatorname{mean}\left((u_x n_x+u_y n_y-h)^2\right).
$$

The normal direction matters. On the unit square's left edge, $n=(-1,0)$, so
the outward derivative is $-u_x$; on its right edge it is $u_x$. The current
Laplace tutorial uses only Dirichlet boundaries. This formula describes how
to extend that loss for a problem with prescribed normal derivatives.

## Balance the constraints and check the result

LearnPDEs currently combines its losses as

$$
\mathcal L=3\mathcal L_{\mathrm{physics}}+\mathcal L_{\mathrm{boundary}}.
$$

Treat this weighting as an experiment setting. PDE residuals, values, and
derivatives can have different scales and units. If a constraint is poorly
satisfied, inspect its loss and physical error before adjusting weights;
increasing its weight may trade away accuracy in another term. Rescaling or
nondimensionalizing a new problem can also make the terms easier to balance.

Run the full examples to observe how the combined loss affects the solution:

```bash
uv run python -m examples.train_pinn cosinus --epochs 5000 --points 64
uv run python -m examples.train_pinn laplace --epochs 5000 --points 21
```

![PINN training with value and derivative initial conditions for the cosine ODE](../../assets/cosinus.gif)

The existing cosine animation illustrates an earlier run. Use the commands'
printed RMSE and maximum error to evaluate your own run, and use the snippets
above as patterns for inspecting individual constraints on your trained model.

Common implementation checks:

- Keep predictions and targets the same shape. Subtracting `(M,)` from `(M, 1)`
  broadcasts into an `(M, M)` array, changing the loss.
- Keep gradient tracking enabled for derivative losses. Detaching predictions
  or using `torch.no_grad()` before taking derivatives breaks the computation.
- Recompute predictions after each optimizer update; stale predictions do not
  describe the current model.
- Measure errors at extra points between training coordinates. A fit at sampled
  points alone does not establish the accuracy of the full solution.

See the [Laplace tutorial](laplace-pinn-pytorch.md) for all four boundary terms,
or the [ODE tutorial](ode-pinn-pytorch.md) for the first- and second-order residuals.

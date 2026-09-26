# Physics-informed neural networks for ODEs: a worked example

A physics-informed neural network (PINN) can approximate an ordinary
differential equation by minimizing its residual and enforcing initial
conditions. In this PyTorch tutorial, you will solve exponential growth and
cosine oscillations using LearnPDEs, then compare the network with the exact
solutions.

[All tutorials and setup](README.md) · [Runnable source](../../examples/train_pinn.py)

## Run the examples

After following the [installation instructions](README.md#install-and-run), run
these commands from the repository root:

```bash
uv run python -m examples.train_pinn exponential --epochs 5000 --points 64
uv run python -m examples.train_pinn cosinus --epochs 5000 --points 64
```

The network has one input coordinate, four hidden layers of 20 `Tanh` units,
and one scalar output. LearnPDEs samples 64 evenly spaced coordinates on
`[-3, 3]` and adds `x = 0` explicitly so the initial conditions can be evaluated.
These coordinates are called **collocation points**. No measured solution
values are needed, but the equations and initial conditions are still inputs
to the problem.

## Exponential growth: a first-order ODE

The first problem is

$$
f'(x)=f(x),\qquad f(0)=1.
$$

Its exact solution is $f(x)=e^x$. Replace the unknown function with a neural
network $f_\theta(x)$, where $\theta$ denotes its trainable parameters. At each
collocation point, the residual is

$$
r_\theta(x)=f_\theta'(x)-f_\theta(x).
$$

The two parts of the loss are

$$
\mathcal L_{\mathrm{physics}}=\frac{1}{N}\sum_{i=1}^{N}r_\theta(x_i)^2,
\qquad
\mathcal L_{\mathrm{initial}}=(f_\theta(0)-1)^2.
$$

The residual alone permits every function $C e^x$, including the zero function.
The initial condition selects the desired amplitude $C=1$. LearnPDEs calls
this initial-condition term `boundary_loss` in its shared loss API.

![LearnPDEs PINN training toward the exponential solution of f prime equals f](../../assets/exponential.gif)

This existing animation illustrates training toward the analytical solution.
It comes from an earlier run; the tutorial command reports numerical errors
and does not regenerate the animation.

## Cosine oscillations: a second-order ODE

The next problem needs two initial conditions:

$$
f''(x)=-f(x),\qquad f(0)=1,\qquad f'(0)=0.
$$

The exact solution is $f(x)=\cos(x)$. Its residual and initial-condition loss are

$$
r_\theta(x)=f_\theta''(x)+f_\theta(x),
$$

$$
\mathcal L_{\mathrm{initial}}=(f_\theta(0)-1)^2+(f_\theta'(0))^2.
$$

The **plus sign in the residual** matters. The equation $f''=f$ with the same
initial conditions gives $\cosh(x)$, not $\cos(x)$. Without the derivative
condition, $\cos(x)+B\sin(x)$ would also satisfy the equation and value at zero.

![LearnPDEs PINN training toward the cosine solution of f double prime plus f equals zero](../../assets/cosinus.gif)

## Compute the derivatives and losses in PyTorch

Save the following as `inspect_ode.py` in the repository root and run
`uv run python inspect_ode.py`. It evaluates both loss functions at their
initial network parameters; the training commands above perform optimization.

```python
import torch

from examples.train_pinn import build_problem

for scenario in ('exponential', 'cosinus'):
    torch.manual_seed(0)
    model, problem, _ = build_problem(scenario, points=64)
    f = model(problem.inputs)
    df = problem.partial_derivative(f, problem.x)

    if scenario == 'exponential':
        residual = df - f
        boundary_loss = (f[problem.zero_mask] - 1).square().mean()
    else:
        ddf = problem.partial_derivative(df, problem.x)
        residual = ddf + f
        boundary_loss = (f[problem.zero_mask] - 1).square().mean() + df[
            problem.zero_mask
        ].square().mean()

    physics_loss = residual.square().mean()
    total_loss = 3 * physics_loss + boundary_loss
    reference_loss, _, _, _ = problem.get_loss(scenario)()
    torch.testing.assert_close(total_loss, reference_loss)
    print(scenario, 'physics:', physics_loss.item())
    print(scenario, 'initial conditions:', boundary_loss.item())
```

[`Loss.partial_derivative`](../../learnpdes/model/loss.py) calls
`torch.autograd.grad` with `create_graph=True`. This records operations used to
compute derivatives, enabling second derivatives and backpropagation through
the physics loss. The `grad_outputs=torch.ones_like(f)` argument computes a
vector-Jacobian product; for this network, each row's output depends only on
that row's coordinate, giving the needed per-point derivatives.
[PyTorch autograd reference](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)

`Tanh` provides smooth derivatives for these examples. A piecewise linear
activation such as ReLU has a zero second derivative almost everywhere, making
it a poor default for this second-order residual.

## Train and check the solution

The [runner](../../examples/train_pinn.py) repeatedly evaluates the chosen
loss, backpropagates it, and updates the parameters with Adam. It uses
`3 * physics_loss + boundary_loss`, matching the existing implementation.
The factor `3` is a project choice, not a universal PINN weighting rule.

After training, inspect the final RMSE and maximum absolute error against
`exp(x)` or `cos(x)` on a separate 201-point evaluation grid. Lower is better;
compare with the initial RMSE rather than interpreting the training loss as
solution error. For exponential growth, an absolute error near `x = 3` can
contribute strongly because the function is much larger there.

If accuracy is insufficient, increase the epochs, compare another `--seed`,
and inspect the initial-condition loss separately using the pattern above.
The model is trained on `[-3, 3]`; this experiment does not establish accuracy
outside that interval.

Continue with [the Laplace equation in two dimensions](laplace-pinn-pytorch.md)
or [boundary-condition losses](pinn-boundary-condition-losses-pytorch.md).

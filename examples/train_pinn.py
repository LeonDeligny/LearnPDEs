"""Train the tutorial scenarios with LearnPDEs and report numerical error.

From the repository root:
    uv run python -m examples.train_pinn laplace --epochs 5000 --points 21
"""

import argparse
from collections.abc import Callable

import torch

from learnpdes import (
    COSINUS_SCENARIO,
    EXPONENTIAL_SCENARIO,
    LAPLACE_SCENARIO,
    device,
)
from learnpdes.model.loss import Loss
from learnpdes.model.pinn import PINN
from learnpdes.utils.loadscenarios import load_scenario


def build_problem(scenario: str, points: int) -> tuple[PINN, Loss, Callable]:
    """Use the same model, collocation points, and losses as learnpdes.main."""
    (
        input_space,
        mesh_masks,
        output_dim,
        analytical,
        input_homeo,
        output_homeo,
        encoding,
    ) = load_scenario(scenario, num_inputs=points)
    input_dim = 1 if input_space.ndim == 1 else input_space.shape[1]
    model = PINN(
        nn_params={
            'input_dim': input_dim,
            'hidden_dim': 20,
            'output_dim': output_dim,
            'num_hidden_layers': 4,
            'activation': torch.nn.Tanh,
        },
        input_homeo=input_homeo,
        output_homeo=output_homeo,
        encoding=encoding,
    ).to(device)
    objective = Loss(
        scenario=scenario,
        input_space=input_space,
        input_dim=input_dim,
        forward=model.forward,
        mesh_masks=mesh_masks,
    )
    return model, objective, analytical


def evaluate(model: PINN, scenario: str, analytical: Callable) -> dict[str, float]:
    """Compare predictions with the exact solution on a separate grid."""
    if scenario == LAPLACE_SCENARIO:
        axis = torch.linspace(0, 1, 41)
        coordinates = torch.cartesian_prod(axis, axis)
    else:
        coordinates = torch.linspace(-3, 3, 201).view(-1, 1)

    # The analytical solution is used only for evaluation, never for training.
    exact = analytical(*(column.numpy() for column in coordinates.unbind(dim=1)))
    exact = torch.as_tensor(exact, dtype=coordinates.dtype, device=device).view(-1, 1)
    model.eval()
    with torch.no_grad():
        error = model(coordinates.to(device)) - exact
        return {
            'rmse': error.square().mean().sqrt().item(),
            'max_error': error.abs().max().item(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'scenario',
        choices=[EXPONENTIAL_SCENARIO, COSINUS_SCENARIO, LAPLACE_SCENARIO],
    )
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument(
        '--points',
        type=int,
        help='Points per axis (default: 256 for exponential, 64 for cosinus, 21 for Laplace).',
    )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    points = args.points
    if points is None:
        points = {
            EXPONENTIAL_SCENARIO: 256,
            COSINUS_SCENARIO: 64,
            LAPLACE_SCENARIO: 21,
        }[args.scenario]
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    if points < 3:
        parser.error('--points must be at least 3')

    torch.manual_seed(args.seed)
    model, objective, analytical = build_problem(args.scenario, points)
    loss_function = objective.get_loss(args.scenario)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    initial = evaluate(model, args.scenario, analytical)
    print(f'Initial RMSE: {initial["rmse"]:.6e}')
    model.train()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss, _, _, _ = loss_function()
        if not torch.isfinite(total_loss):
            raise RuntimeError(f'Non-finite loss at epoch {epoch}')
        # Loss caches coordinate graphs and the Laplace boundary target.
        # Match Trainer.train's graph handling when reusing this Loss object.
        total_loss.backward(retain_graph=True)
        optimizer.step()
        if epoch == 1 or epoch % max(1, args.epochs // 5) == 0:
            print(f'Epoch {epoch}/{args.epochs}: loss={total_loss.item():.6e}')

    final_loss, _, _, _ = loss_function()
    final = evaluate(model, args.scenario, analytical)
    print(f'Final loss: {final_loss.item():.6e}')
    print(f'Final RMSE: {final["rmse"]:.6e}')
    print(f'Maximum absolute error: {final["max_error"]:.6e}')


if __name__ == '__main__':
    main()

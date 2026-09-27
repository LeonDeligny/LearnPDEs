"""Shared problem construction, evaluation, and artifact-producing training."""

from collections.abc import Callable

import numpy as np
import torch

from learnpdes import (
    COSINUS_SCENARIO,
    CYLINDER_SCENARIO,
    KOVASZNAY_SCENARIO,
    LAPLACE_SCENARIO,
    POISEUILLE_SCENARIO,
    cosinus,
    poiseuille,
    device,
)
from learnpdes.fluid import (
    CylinderProblem,
    KovasznayProblem,
    FluidObjective,
    build_fluid_problem,
)
from learnpdes.fluid_evaluation import evaluate_fluid
from learnpdes.model.loss import Loss
from learnpdes.model.pinn import PINN
from learnpdes.model.trainer import Trainer
from learnpdes.scenarios import (
    RunConfig,
    WIND_TUNNEL_SCENARIO,
    get_scenario,
)
from learnpdes.utils.artifacts import TrainingRun, problem_settings
from learnpdes.utils.loadscenarios import load_scenario, load_2d_mesh, load_wind_tunnel
from learnpdes.utils.plot import get_plot_func, require_gif_export
from learnpdes.utils.visualization import (
    ModelEvaluator,
    VisualizationGrid,
    visualization_grid,
    rectangular_triangles,
)


def build_problem(
    scenario: str,
    points: int,
    *,
    cosinus_order: int = 2,
    hidden_dim: int | None = None,
    hidden_layers: int = 4,
    mesh_path=None,
) -> tuple[PINN, Loss | FluidObjective, Callable | None]:
    """Construct every registered case using the same defaults as the CLI."""
    config = RunConfig(
        scenario,
        points=points,
        cosinus_order=cosinus_order,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        mesh_path=mesh_path,
    ).resolved()
    case = get_scenario(config.scenario)
    if case.fluid:
        return build_fluid_problem(
            case.name,
            points,
            hidden_dim=config.hidden_dim,
            hidden_layers=hidden_layers,
        )
    if case.mesh:
        data = load_2d_mesh(points, filepath=config.mesh_path)
    elif case.name == WIND_TUNNEL_SCENARIO:
        data = load_wind_tunnel(points)
        data[1]['airfoil'] = torch.zeros(len(data[0]), dtype=torch.bool)
    else:
        data = load_scenario(case.physics, num_inputs=points)
    inputs, masks, outputs, analytical, input_homeo, output_homeo, encoding = data
    input_dim = 1 if inputs.ndim == 1 else inputs.shape[1]
    model = PINN(
        nn_params={
            'input_dim': input_dim,
            'hidden_dim': config.hidden_dim,
            'output_dim': outputs,
            'num_hidden_layers': hidden_layers,
            'activation': torch.nn.Tanh,
        },
        input_homeo=input_homeo,
        output_homeo=output_homeo,
        encoding=encoding,
    ).to(device)
    objective = Loss(
        scenario=case.physics,
        input_space=inputs,
        input_dim=input_dim,
        forward=model.forward,
        mesh_masks=masks,
        cosinus_order=cosinus_order,
        cosinus_derivatives=model.input_derivatives if cosinus_order >= 8 else None,
    )
    return model, objective, analytical


def evaluate(
    model: PINN, scenario: str, analytical: Callable | None
) -> dict[str, float]:
    """Compare with exact formulas or report physics diagnostics; no simulation data."""
    if scenario in (CYLINDER_SCENARIO, KOVASZNAY_SCENARIO):
        problem = (
            CylinderProblem() if scenario == CYLINDER_SCENARIO else KovasznayProblem()
        )
        return evaluate_fluid(model, problem)
    if analytical is None:
        return {}
    if scenario == LAPLACE_SCENARIO:
        axis = torch.linspace(0, 1, 41)
        coordinates = torch.cartesian_prod(axis, axis)
    elif scenario == POISEUILLE_SCENARIO:
        coordinates = torch.cartesian_prod(
            torch.linspace(0, poiseuille.LENGTH, 41),
            torch.linspace(0, poiseuille.HEIGHT, 41),
        )
    else:
        bounds = cosinus.EVALUATION_BOUNDS if scenario == COSINUS_SCENARIO else (-3, 3)
        count = cosinus.EVALUATION_POINTS if scenario == COSINUS_SCENARIO else 201
        coordinates = torch.linspace(*bounds, count).view(-1, 1)

    # Interior exact values are used only for evaluation. Kovasznay training
    # uses exact velocity boundary values and one pressure reference.
    exact = analytical(*(column.numpy() for column in coordinates.unbind(dim=1)))
    exact = (
        np.column_stack(exact)
        if isinstance(exact, tuple)
        else np.asarray(exact)[:, None]
    )
    exact = torch.as_tensor(exact, dtype=coordinates.dtype, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        try:
            prediction = model(coordinates.to(device))
        finally:
            model.train(was_training)
        error = prediction - exact
        metrics = {
            'rmse': error.square().mean().sqrt().item(),
            'max_error': error.abs().max().item(),
            'relative_l2': (
                torch.linalg.vector_norm(error) / torch.linalg.vector_norm(exact)
            ).item(),
        }
        if scenario == POISEUILLE_SCENARIO:
            for index, name in enumerate(('u', 'v', 'p')):
                metrics[f'{name}_rmse'] = error[:, index].square().mean().sqrt().item()
                metrics[f'{name}_max_error'] = error[:, index].abs().max().item()
                # The Poiseuille v reference is zero, so use absolute errors.
                if torch.count_nonzero(exact[:, index]):
                    metrics[f'{name}_relative_l2'] = (
                        torch.linalg.vector_norm(error[:, index])
                        / torch.linalg.vector_norm(exact[:, index])
                    ).item()
        if scenario == COSINUS_SCENARIO:
            metrics.update(
                {
                    f'{region}_mse': value
                    for region, value in cosinus.region_mse(
                        coordinates.numpy(), prediction.cpu().numpy()
                    ).items()
                }
            )
        return metrics


def train(config: RunConfig) -> Trainer:
    """Run one validated case and retain its resolved settings and artifacts."""
    config = config.resolved()
    case = get_scenario(config.scenario)
    if config.save_gif:
        require_gif_export()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.threads)
    model, objective, analytical = build_problem(
        case.name,
        config.points,
        cosinus_order=config.cosinus_order,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        mesh_path=config.mesh_path,
    )
    if case.name == WIND_TUNNEL_SCENARIO:
        loss_function = objective.get_pre_loss(case.physics)
        x, y = np.meshgrid(
            np.linspace(0, 4, config.resolution),
            np.linspace(0, 1, config.resolution),
            indexing='ij',
        )
        grid = VisualizationGrid(np.column_stack((x.ravel(), y.ravel())))
        grid.triangles = rectangular_triangles(grid.coordinates)
    else:
        loss_function = objective.get_loss(case.physics)
        grid_options = {'mesh_path': config.mesh_path} if case.mesh else {}
        grid = visualization_grid(
            case.physics,
            objective.input_space,
            config.resolution,
            **grid_options,
        )
    has_validation = analytical is not None or case.fluid
    if has_validation:
        initial = evaluate(model, case.name, analytical)
        if 'rmse' in initial:
            print(f'Initial RMSE: {initial["rmse"]:.6e}')
    model.train()
    run = TrainingRun(
        config.output_dir,
        case.name,
        settings=problem_settings(
            model,
            objective,
            seed=config.seed,
            points_per_axis=config.points,
            visualization_resolution=config.resolution,
            cosinus_order=config.cosinus_order,
            config=config.as_dict(),
        ),
    )
    trainer = Trainer(
        model.parameters,
        loss_function,
        {
            'learning_rate': config.learning_rate,
            'epochs': config.epochs,
            'lbfgs_steps': config.lbfgs_steps,
            'resample_every': config.resample_every,
        },
        {
            'plot_func': get_plot_func(
                case.physics, cosinus_order=config.cosinus_order
            ),
            'evaluate': ModelEvaluator(
                model, case.physics, grid, density=objective.rho.item()
            ),
            **run.plot_options(gif=config.save_gif),
            'max_frames': config.max_frames,
        },
        analytical=analytical,
        run=run,
        model=model,
        validation=(lambda: evaluate(model, case.name, analytical))
        if has_validation
        else None,
        objective=objective if isinstance(objective, FluidObjective) else None,
    )
    trainer.train()
    print(f'Final loss: {trainer.loss_history[-1][1]:.6e}')
    final = trainer.validation_result or {}
    labels = {
        'rmse': 'Final RMSE',
        'max_error': 'Maximum absolute error',
        'relative_l2': 'Relative L2 error',
    }
    for name, value in final.items():
        print(f'{labels.get(name, name)}: {value:.6e}')
    return trainer

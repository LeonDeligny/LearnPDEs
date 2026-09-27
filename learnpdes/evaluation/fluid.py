"""Independent fluid diagnostics and verification gates, parameterized by a scenario."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from learnpdes.model.fluid import FluidProblem


@dataclass(frozen=True)
class FluidEvaluation:
    pressure_gauge: str
    analytical: (
        Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, ...]] | None
    ) = None
    implementation: str | None = None
    formula: str | None = None
    source: str | None = None
    hard_boundaries: bool = False
    observables: Callable[..., dict[str, float]] | None = None
    diagnostic_limits: tuple[tuple[str, float], ...] = ()

    def evaluate(
        self,
        model: torch.nn.Module,
        problem: FluidProblem,
        *,
        count: int = 2048,
        seed: int = 1729,
    ) -> dict[str, float]:
        """Unseen uniform interior points, independent boundaries, no gauge fitting."""
        parameter = next(model.parameters())
        generator = torch.Generator().manual_seed(seed)
        samples = problem.sample(count, 256, generator=generator, near_cylinder=False)
        was_training = model.training
        model.eval()
        try:
            metrics = {}
            sums = {'continuity': 0.0, 'momentum_u': 0.0, 'momentum_v': 0.0}
            prediction_batches = []
            with torch.enable_grad():
                for batch in samples.interior.split(256):
                    xy = batch.to(parameter).detach().requires_grad_(True)
                    fields = model(xy)
                    prediction_batches.append(fields.detach())
                    for name, residual in problem.residuals(xy, fields).items():
                        sums[name] += residual.detach().square().sum().item()
                metrics.update(
                    {
                        f'{name}_rms': math.sqrt(value / count)
                        for name, value in sums.items()
                    }
                )
                boundary = {
                    name: xy.to(parameter).detach().requires_grad_(True)
                    for name, xy in samples.boundary.items()
                }
                for name, residual in problem.boundary_residuals(
                    model, boundary
                ).items():
                    metrics[f'{name}_rms'] = (
                        residual.detach().square().mean().sqrt().item()
                    )
                if self.observables is not None:
                    metrics.update(self.observables(model, problem))
                if self.analytical is None:
                    # Without an admissible reference, report only physics diagnostics.
                    return metrics
            xy = samples.interior.to(parameter)
            with torch.no_grad():
                reference = torch.stack(self.analytical(*xy.unbind(1)), dim=1)
                error = torch.cat(prediction_batches) - reference
                metrics.update(
                    {
                        'rmse': error.square().mean().sqrt().item(),
                        'max_error': error.abs().max().item(),
                        'relative_l2': (error.norm() / reference.norm()).item(),
                    }
                )
                for index, name in enumerate(('u', 'v', 'p')):
                    metrics[f'{name}_rmse'] = (
                        error[:, index].square().mean().sqrt().item()
                    )
                    metrics[f'{name}_max_error'] = error[:, index].abs().max().item()
                    metrics[f'{name}_relative_l2'] = (
                        error[:, index].norm() / reference[:, index].norm()
                    ).item()
            return metrics
        finally:
            model.train(was_training)

    @staticmethod
    def _reference_checks(metrics: dict[str, float]) -> dict[str, bool]:
        return {
            f'{name}_{metric}': metrics[f'{name}_{metric}'] <= limit
            for name in ('u', 'v', 'p')
            for metric, limit in (('relative_l2', 0.01), ('max_error', 0.02))
        }

    def _checks(
        self, metrics: dict[str, float], hard_tolerance: float
    ) -> dict[str, bool]:
        exact = self.analytical is not None
        checks = {
            f'{name}_rms': metrics[f'{name}_rms'] <= 0.01
            for name in ('continuity', 'momentum_u', 'momentum_v')
        }
        checks.update(
            {
                name: value <= 0.001
                for name, value in metrics.items()
                if name.endswith('_rms') and name not in checks
            }
        )
        if self.hard_boundaries:
            for name, value in metrics.items():
                if name.endswith(('_u_rms', '_v_rms')) and name not in (
                    'momentum_u_rms',
                    'momentum_v_rms',
                ):
                    checks[name] = value <= hard_tolerance
        if exact:
            checks.update(self._reference_checks(metrics))
        checks.update(
            {name: metrics[name] <= limit for name, limit in self.diagnostic_limits}
        )
        return checks

    def report(
        self,
        model: torch.nn.Module,
        problem: FluidProblem,
        *,
        training_seed: int | None = None,
    ) -> dict[str, Any]:
        """Log a single run against fixed checks; never call this full acceptance.

        The project acceptance gate additionally requires all three training seeds
        and two collocation densities. Reference availability is scenario-specific.
        """
        exact = self.analytical is not None
        hard_tolerance = 32 * torch.finfo(next(model.parameters()).dtype).eps
        evaluations = []
        for count, seed in ((4096, 2027), (8192, 1729)):
            metrics = self.evaluate(model, problem, count=count, seed=seed)
            checks = self._checks(metrics, hard_tolerance)
            evaluations.append(
                {
                    'interior_count': count,
                    'boundary_count_per_edge': 256,
                    'seed': seed,
                    'metrics': metrics,
                    'checks': checks,
                    'passed': all(checks.values()),
                }
            )
        return {
            'scenario': problem.name,
            'training_seed': training_seed,
            'reference_type': 'analytical' if exact else 'none',
            'reference_implementation': self.implementation,
            'reference_formula': self.formula,
            'reference_source': self.source,
            'pressure_gauge': self.pressure_gauge,
            'thresholds': {
                'pde_rms': 0.01,
                'boundary_rms': 0.001,
                'hard_boundary_rms': hard_tolerance,
                'relative_l2': 0.01 if exact else None,
                'max_absolute_error': 0.02 if exact else None,
                'characteristic_output_scales': [1, 1, 1],
            },
            'evaluations': evaluations,
            'single_run_checks_passed': all(item['passed'] for item in evaluations),
            'project_acceptance': False,
            'limitations': 'Single training seed and density; the multi-seed, two-density project gate is not established.'
            if exact
            else 'No admissible exact solution for this setup. Residuals and flux do not bound full-field error.',
        }

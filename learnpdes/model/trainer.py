"""Training and visualization checkpoints for PINN models."""

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from torch import Tensor
from torch.nn import Module
from torch.optim import LBFGS, Adam, Optimizer

from learnpdes.model.fluid import FluidObjective
from learnpdes.types import (
    Analytical,
    FieldEvaluation,
    LossFunction,
    LossHistory,
    LossResult,
    PlotOptions,
    PointGroups,
    TrainingParams,
)
from learnpdes.utils.artifacts import TrainingRun
from learnpdes.utils.collocation import CollocationRecorder
from learnpdes.utils.plot import create_gif, require_gif_export
from learnpdes.utils.utility import detach_to_numpy


def checkpoint_steps(epochs: int, max_frames: int = 80) -> set[int]:
    """Spend more frames on rapid early changes and always include both endpoints."""
    if epochs < 0:
        raise ValueError('Epoch count cannot be negative.')
    if max_frames < 2:
        raise ValueError('At least two checkpoint frames are required.')
    if epochs == 0:
        return {0}
    if max_frames == 2:
        return {0, epochs}
    samples = np.geomspace(1, epochs, num=min(epochs, max_frames - 1))
    return {0, epochs, *(int(round(step)) for step in samples)}


class Trainer:
    def __init__(
        self,
        model_params: Callable[[], Iterable[Tensor]],
        loss: LossFunction,
        training_params: TrainingParams,
        plot: PlotOptions,
        analytical: Analytical | None = None,
        *,
        run: TrainingRun | None = None,
        model: Module | None = None,
        validation: Callable[[], dict[str, float]] | None = None,
        objective: FluidObjective | None = None,
        collocation: Callable[[], PointGroups] | None = None,
    ) -> None:
        """Bind training callbacks, optimizer settings, and checkpoint exports."""
        self.loss = loss
        self.run = run
        self.model = model
        self.validation = validation
        self.validation_result: dict[str, float] | None = None
        self.objective = objective
        self.collocation = collocation or getattr(
            objective or getattr(loss, '__self__', None), 'collocation_points', None
        )
        self.collocation_history = CollocationRecorder()
        self.resample_every = training_params.get('resample_every', 100)
        self.lbfgs_steps = training_params.get('lbfgs_steps', 0)
        if self.resample_every < 1 or self.lbfgs_steps < 0:
            raise ValueError('Invalid sampling interval or L-BFGS step count.')
        if self.lbfgs_steps and objective is None:
            raise ValueError(
                'L-BFGS requires a fluid objective with fresh coordinate graphs.'
            )
        self.component_history: list[dict[str, object]] = []
        self.model_params = list(model_params())
        self.completed_steps = 0
        self.learning_rate = training_params['learning_rate']
        self.nb_epochs = training_params['epochs']
        if self.nb_epochs < 0:
            raise ValueError('Epoch count cannot be negative.')
        self.total_steps = self.nb_epochs + self.lbfgs_steps
        self.checkpoints = checkpoint_steps(
            self.total_steps, plot.get('max_frames', 80)
        )
        self.plot_func = plot['plot_func']
        self.evaluate = plot.get('evaluate')
        self.output_dir = Path(plot.get('output_dir', './assets'))
        self.frame_dir = Path(plot.get('frame_dir', self.output_dir))
        html_path = plot.get(
            'html_path',
            self.output_dir / 'training.html'
            if hasattr(self.plot_func, 'write_html')
            else None,
        )
        self.html_path = Path(html_path) if html_path is not None else None
        gif_path = plot.get('gif_path')
        self.gif_path = Path(gif_path) if gif_path is not None else None
        self.duration_ms = plot.get('duration_ms', 100)
        self.final_hold_ms = plot.get('final_hold_ms', 2000)
        self.loss_history: LossHistory = []
        self.analytical = analytical
        self.optimizer: Optimizer = Adam(self.model_params, lr=self.learning_rate)

    def train(self) -> None:
        if self.run is not None:
            if self.run.manifest['status'] != 'created':
                raise ValueError(
                    'Create a new TrainingRun for each training invocation.'
                )
            self.run.update(
                status='training',
                training={
                    'optimizer': 'Adam + L-BFGS' if self.lbfgs_steps else 'Adam',
                    'learning_rate': self.learning_rate,
                    'epochs': self.nb_epochs,
                    'lbfgs_steps': self.lbfgs_steps,
                    'resample_every': self.resample_every if self.objective else None,
                },
                export={
                    'gif': self.gif_path is not None,
                    'checkpoint_steps': sorted(self.checkpoints),
                    'duration_ms': self.duration_ms,
                    'final_hold_ms': self.final_hold_ms,
                },
            )
        try:
            self._train()
        except BaseException as error:
            if self.run is not None:
                try:
                    self.collocation_history.save(
                        self.run.directory / 'collocation.json'
                    )
                    self.run.save_training(
                        self.loss_history,
                        self.model,
                        self.optimizer,
                        self.completed_steps,
                        components=self.component_history,
                    )
                    self.run.update(
                        status='interrupted'
                        if isinstance(error, KeyboardInterrupt)
                        else 'failed',
                        completed_steps=self.completed_steps,
                        error={'type': type(error).__name__, 'message': str(error)},
                    )
                except Exception as save_error:
                    error.add_note(f'Could not save the failure record: {save_error}')
            raise

    def _train(self) -> None:
        if self.gif_path is not None and hasattr(self.plot_func, 'write_frames'):
            require_gif_export()
        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.plot_func, 'reset'):
            getattr(self.plot_func, 'reset')()
        self.loss_history.clear()
        self.component_history.clear()
        self.collocation_history = CollocationRecorder()
        # Step N refers to exactly N completed optimizer updates. Evaluating at
        # the beginning of the next iteration keeps fields and loss in sync.
        for step in range(self.total_steps + 1):
            self._prepare_step(step)
            self.optimizer.zero_grad()
            result = self.loss()
            loss = result[0]
            loss_value = float(loss.item())
            if not np.isfinite(loss_value):
                raise RuntimeError(f'Non-finite loss at step {step}')
            self.loss_history.append((step, loss_value))
            self._record_components(step)
            if step in self.checkpoints:
                self._capture_checkpoint(step, result)
            if step == self.total_steps:
                break
            if isinstance(self.optimizer, LBFGS):
                # Release the evaluated graph before the fresh line-search closures.
                del loss, result
                self.optimizer.step(self._lbfgs_closure)
            else:
                loss.backward(retain_graph=self.objective is None)
                self.optimizer.step()
            self.completed_steps = step + 1
        self._export()

    def _prepare_step(self, step: int) -> None:
        if self.objective is not None and (
            step == 0
            or (step < self.nb_epochs and step % self.resample_every == 0)
            or (step == self.nb_epochs and self.lbfgs_steps)
        ):
            self.objective.resample()
        if step == self.nb_epochs and self.lbfgs_steps:
            # One outer update per recorded step. Every line-search closure
            # uses the same coordinates throughout the entire L-BFGS phase.
            self.optimizer = LBFGS(
                self.model_params,
                lr=1.0,
                max_iter=1,
                max_eval=25,
                history_size=100,
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
                line_search_fn='strong_wolfe',
            )

    def _record_components(self, step: int) -> None:
        if self.objective is not None:
            self.component_history.append(
                {
                    'step': step,
                    'phase': 'adam'
                    if step < self.nb_epochs or not self.lbfgs_steps
                    else 'lbfgs',
                    **self.objective.components,
                }
            )

    def _capture_checkpoint(self, step: int, result: LossResult) -> None:
        loss, inputs, values, geometry_mask = result
        samples = self.collocation_history.capture(
            step,
            self.collocation()
            if self.collocation
            else {'equation': {'kind': 'pde', 'coordinates': inputs}},
        )
        print(f'Step {step}, Loss: {loss.item():.6e}', flush=True)
        if self.objective is not None:
            print(
                ', '.join(
                    f'{name}={value:.3e}'
                    for name, value in self.objective.components.items()
                )
            )
        evaluation: FieldEvaluation = (
            self.evaluate()
            if self.evaluate is not None
            else {
                'inputs': detach_to_numpy(inputs),
                'f': detach_to_numpy(values),
                'geometry_mask': (
                    detach_to_numpy(geometry_mask)
                    if geometry_mask is not None
                    else None
                ),
            }
        )
        self.plot_func(
            self.output_dir,
            epoch=step,
            loss=float(loss.item()),
            analytical=self.analytical,
            loss_history=self.loss_history,
            total_epochs=self.total_steps,
            collocation=samples,
            **evaluation,
        )

    def _lbfgs_closure(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        value, *_ = self.loss()
        if not np.isfinite(value.item()):
            raise RuntimeError('Non-finite loss in L-BFGS closure')
        value.backward()
        return float(value.item())

    def _export(self) -> None:
        if self.run is not None:
            self.collocation_history.save(self.run.directory / 'collocation.json')
            self.run.save_training(
                self.loss_history,
                self.model,
                self.optimizer,
                self.completed_steps,
                components=self.component_history,
            )
            self.run.update(status='exporting', completed_steps=self.completed_steps)
        if self.validation is not None:
            self.validation_result = self.validation()
            if self.run is not None:
                self.run.update(validation=self.validation_result)
        if self.html_path is not None:
            getattr(self.plot_func, 'write_html')(self.html_path)
            print(f'Saved interactive training figure: {self.html_path}')
        # Render after the last update, with fixed ranges across the entire run.
        if self.gif_path is not None:
            print(f'Rendering {len(self.checkpoints)} training checkpoints for GIF...')
            if hasattr(self.plot_func, 'write_frames'):
                getattr(self.plot_func, 'write_frames')(self.frame_dir)
            create_gif(
                self.gif_path,
                self.frame_dir,
                duration_ms=self.duration_ms,
                final_hold_ms=self.final_hold_ms,
            )
            print(f'Saved training GIF: {self.gif_path}')
        if self.run is not None:
            self.run.finish(final_loss=self.loss_history[-1][1])
            print(f'Run record: {self.run.directory / "run.json"}')

"""Training and visualization checkpoints for PINN models."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from torch.optim import Adam

from learnpdes.utils.utility import detach_to_numpy
from learnpdes.utils.plot import create_gif, ensure_directory_exists


def checkpoint_steps(epochs: int, max_frames: int = 80) -> set[int]:
    """Spend more frames on rapid early changes and always include both endpoints."""
    if epochs < 0:
        raise ValueError('Epoch count cannot be negative.')
    if max_frames < 2:
        raise ValueError('At least two checkpoint frames are required.')
    if epochs == 0:
        return {0}
    samples = np.geomspace(1, epochs, num=min(epochs, max_frames - 1))
    return {0, epochs, *(int(round(step)) for step in samples)}


class Trainer:
    def __init__(
        self,
        model_params: Callable,
        loss: Callable,
        training_params: dict,
        plot: dict,
        analytical: Callable | None = None,
    ) -> None:
        self.loss = loss
        self.learning_rate = training_params['learning_rate']
        self.nb_epochs = training_params['epochs']
        self.checkpoints = checkpoint_steps(self.nb_epochs, plot.get('max_frames', 80))
        self.plot_func = plot['plot_func']
        self.evaluate = plot.get('evaluate')
        self.output_dir = Path(plot.get('output_dir', './gifs/epochs'))
        self.gif_path = Path(plot.get('gif_path', './gifs/training_process.gif'))
        self.duration_ms = plot.get('duration_ms', 100)
        self.final_hold_ms = plot.get('final_hold_ms', 2000)
        self.loss_history = []
        self.analytical = analytical
        self.optimizer = Adam(model_params(), lr=self.learning_rate)

    def train(self) -> None:
        if self.nb_epochs == 0:
            return
        output_dir = ensure_directory_exists(self.output_dir)
        self.loss_history.clear()
        # Step N refers to exactly N completed optimizer updates. Evaluating at
        # the beginning of the next iteration keeps fields and loss in sync.
        for step in range(self.nb_epochs + 1):
            self.optimizer.zero_grad()
            loss, inputs, values, geometry_mask = self.loss()
            self.loss_history.append((step, loss.item()))
            if step in self.checkpoints:
                print(f'Step {step}, Loss: {loss.item():.6e}')
                evaluation = (
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
                    output_dir,
                    epoch=step,
                    loss=loss.item(),
                    analytical=self.analytical,
                    loss_history=self.loss_history,
                    total_epochs=self.nb_epochs,
                    **evaluation,
                )
            if step < self.nb_epochs:
                loss.backward(retain_graph=True)
                self.optimizer.step()
            create_gif(
                self.gif_path,
                output_dir,
                duration_ms=self.duration_ms,
                final_hold_ms=self.final_hold_ms,
            )

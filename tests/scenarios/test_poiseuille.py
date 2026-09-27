"""Verify the viscous benchmark independently of a trained approximation."""

import unittest
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch

from learnpdes import POISEUILLE_SCENARIO, device
from learnpdes.scenarios import poiseuille
from learnpdes.scenarios.poiseuille import Objective, load_poiseuille
from learnpdes.training import build_problem


class ExactFlow(torch.nn.Module):
    """Independent polynomial for the default benchmark, with perturbations."""

    def __init__(
        self,
        *,
        slip: float = 0.0,
        pressure_shift: float = 0.0,
        curvature: float = 1.0,
        reverse: bool = False,
    ) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.slip = slip
        self.pressure_shift = pressure_shift
        self.curvature = curvature
        self.reverse = reverse

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, :1], xy[:, 1:2]
        u = self.curvature * 4 * y * (1 - y) + self.slip
        v = torch.zeros_like(u)
        p = 0.8 * (x if self.reverse else 4 - x) + self.pressure_shift
        return self.scale * torch.cat((u, v, p), dim=1)


class TestPoiseuille(unittest.TestCase):
    def make_loss(self, model: torch.nn.Module) -> Objective:
        coordinates, masks, *_ = load_poiseuille(num_inputs=7)
        model = model.to(device)
        return Objective(POISEUILLE_SCENARIO, coordinates, 2, model.forward, masks)

    def test_channel_masks_and_exact_solution(self) -> None:
        coordinates, masks, outputs, analytical, *_ = load_poiseuille(num_inputs=7)
        self.assertEqual(outputs, 3)
        assert analytical is not None
        self.assertEqual(coordinates.shape, (49, 2))
        self.assertEqual(masks['inlet'].sum().item(), 7)
        self.assertEqual(masks['outlet'].sum().item(), 7)
        self.assertEqual(masks['wall'].sum().item(), 14)
        u, v, p = analytical(*coordinates.numpy().T)
        np.testing.assert_array_equal(u[masks['wall']], 0)
        np.testing.assert_array_equal(v, 0)
        np.testing.assert_allclose(p[masks['inlet']], 3.2)
        np.testing.assert_array_equal(p[masks['outlet']], 0)
        center_u, _, center_p = analytical(2.0, 0.5)
        self.assertAlmostEqual(float(center_u), 1.0)
        self.assertAlmostEqual(float(center_p), 1.6)
        np.testing.assert_allclose(u.reshape(7, 7), u.reshape(7, 7)[:, ::-1], atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            load_poiseuille(num_inputs=2)

    def test_exact_flow_has_zero_residual_and_boundary_loss(self) -> None:
        objective = self.make_loss(ExactFlow())
        total, coordinates, fields, geometry = objective.get_loss(POISEUILLE_SCENARIO)()
        self.assertLess(total.item(), 1e-12)
        self.assertIsNone(geometry)
        np.testing.assert_allclose(
            np.array([field.detach().cpu().numpy().ravel() for field in fields]),
            poiseuille.analytical(*coordinates.detach().cpu().numpy().T),
            atol=1e-6,
        )

    def test_wall_slip_pressure_gauge_and_wrong_forcing_are_penalized(self) -> None:
        # Constant slip and pressure shifts leave the PDE unchanged; only the
        # respective wall/pressure boundary conditions can detect them.
        for changes, expected in (
            ({'slip': 0.5}, 0.25),
            ({'pressure_shift': 0.5}, 0.5),
            ({'curvature': 0.0}, 3 * 0.8**2),
        ):
            with self.subTest(changes=changes):
                total, *_ = self.make_loss(
                    ExactFlow(**cast(dict[str, Any], changes))
                ).poiseuille_loss()
                self.assertAlmostEqual(total.item(), expected, places=5)
        total, *_ = self.make_loss(ExactFlow(reverse=True)).poiseuille_loss()
        self.assertGreater(total.item(), 1.0)

    def test_both_momentum_equations_and_continuity(self) -> None:
        # This field has nonzero convection, diffusion, pressure gradients,
        # and divergence, unlike the exact unidirectional Poiseuille solution.
        def polynomial(xy: torch.Tensor) -> torch.Tensor:
            x, y = xy[:, :1], xy[:, 1:2]
            return torch.cat((x**2 + y**2, x * y, x + y), dim=1)

        coordinates, masks, *_ = load_poiseuille(num_inputs=5)
        objective = Objective(POISEUILLE_SCENARIO, coordinates, 2, polynomial, masks)
        x, y = coordinates.unbind(dim=1)
        expected = sum(
            residual.square().mean()
            for residual in (
                3 * x,
                2 * x**3 + 4 * x * y**2 + 0.6,
                2 * x**2 * y + y**3 + 1,
            )
        )
        with patch.object(objective, 'process', wraps=objective.process) as process:
            objective.poiseuille_loss()
        physics, _ = process.call_args.args
        torch.testing.assert_close(physics.cpu(), expected)

    def test_optimizer_updates_all_three_output_fields(self) -> None:
        torch.manual_seed(0)
        model, objective, _ = build_problem(POISEUILLE_SCENARIO, 7)
        assert isinstance(objective, Objective)
        output_layer = model.network[-1]
        assert isinstance(output_layer, torch.nn.Linear)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = output_layer.weight.detach().clone()
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            total, *_ = objective.poiseuille_loss()
            losses.append(total.item())
            total.backward(retain_graph=True)
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                assert parameter.grad is not None
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        self.assertTrue(np.isfinite(losses).all())
        for old, new in zip(before, output_layer.weight):
            self.assertFalse(torch.equal(old, new))


if __name__ == '__main__':
    unittest.main()

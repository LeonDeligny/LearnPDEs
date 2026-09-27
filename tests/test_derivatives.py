"""Check high-order Taylor derivatives and their parameter gradients independently."""

import unittest

import torch

from examples.train_pinn import build_problem
from learnpdes.model.derivatives import tanh_derivatives


def reverse_derivatives(network, x, order):
    derivatives = [network(x)]
    for _ in range(order):
        derivatives.append(
            torch.autograd.grad(derivatives[-1].sum(), x, create_graph=True)[0]
        )
    return derivatives


class TestTaylorDerivatives(unittest.TestCase):
    def test_derivatives_through_twelve_match_reverse_mode(self):
        torch.manual_seed(5)
        network = torch.nn.Sequential(
            torch.nn.Linear(1, 2), torch.nn.Tanh(), torch.nn.Linear(2, 1)
        ).double()
        x = torch.tensor(
            [[-0.6], [0.1], [0.8]], dtype=torch.float64, requires_grad=True
        )
        expected = reverse_derivatives(network, x, 12)
        actual = tanh_derivatives(network, x, 12)
        for order, (left, right) in enumerate(zip(expected, actual)):
            with self.subTest(order=order):
                torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-10)
        expected_loss = sum(value.square().mean() for value in expected)
        actual_loss = sum(value.square().mean() for value in actual)
        left = torch.autograd.grad(expected_loss, tuple(network.parameters()))
        right = torch.autograd.grad(actual_loss, tuple(network.parameters()))
        for first, second in zip(left, right):
            torch.testing.assert_close(first, second, rtol=1e-8, atol=1e-9)

    def test_composed_tanh_network_and_loss_gradients_match(self):
        torch.manual_seed(3)
        layers = [torch.nn.Linear(1, 3), torch.nn.Tanh()]
        for _ in range(3):
            layers += [torch.nn.Linear(3, 3), torch.nn.Tanh()]
        network = torch.nn.Sequential(*layers, torch.nn.Linear(3, 1)).double()
        x = torch.linspace(-3, 3, 7, dtype=torch.float64)[:, None].requires_grad_()
        expected = reverse_derivatives(network, x, 6)
        actual = tanh_derivatives(network, x, 6)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-12)

        def objective(values):
            loss = 3 * sum(
                (values[n] + values[n - 2]).square().mean() for n in (2, 4, 6)
            )
            return (
                loss
                + (values[0][3] - 1).square().sum()
                + values[1][3].square().sum()
                + (values[4][3] - 1).square().sum()
                + (values[6][3] + 1).square().sum()
            )

        left = torch.autograd.grad(objective(expected), tuple(network.parameters()))
        right = torch.autograd.grad(objective(actual), tuple(network.parameters()))
        for first, second in zip(left, right):
            torch.testing.assert_close(first, second, rtol=1e-9, atol=1e-11)

    def test_high_order_training_stays_differentiable_across_updates(self):
        for order in (8, 10, 12):
            with self.subTest(order=order):
                torch.manual_seed(0)
                model, objective, _ = build_problem('cosinus', 8, cosinus_order=order)
                self.assertIsNotNone(objective.cosinus_derivatives)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                for _ in range(2):
                    optimizer.zero_grad()
                    loss = objective.cosinus_loss()[0]
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward(retain_graph=True)
                    self.assertTrue(
                        all(
                            parameter.grad is not None
                            and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()
                        )
                    )
                    optimizer.step()


if __name__ == '__main__':
    unittest.main()

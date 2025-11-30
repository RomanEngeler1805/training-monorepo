import torch

from training.optimizer import SGD


class TestSGD:
    def test_sgd_init(self):
        """Test that SGD initializes correctly"""
        # Create dummy parameters
        param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        param2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        params = [param1, param2]

        optimizer = SGD(params, lr=0.1, lr_decay=1.01)

        assert optimizer.lr == 0.1
        assert optimizer.lr_decay == 1.01
        assert len(optimizer.model_parameters) == 2

    def test_step_updates_parameters(self):
        """Test that step() updates parameters using gradients"""
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        param.grad = torch.tensor([0.5, -0.3])
        optimizer = SGD([param], lr=0.1, lr_decay=1.0)

        initial_value = param.data.clone()
        optimizer.step()

        # Should update: param = param - lr * grad
        expected = initial_value - 0.1 * param.grad
        torch.testing.assert_close(param.data, expected)

    def test_zero_grad_resets_gradients(self):
        """Test that zero_grad() resets all gradients to zero"""
        param1 = torch.nn.Parameter(torch.tensor([1.0]))
        param2 = torch.nn.Parameter(torch.tensor([2.0]))
        param1.grad = torch.tensor([0.5])
        param2.grad = torch.tensor([-0.3])

        optimizer = SGD([param1, param2], lr=0.1)
        optimizer.zero_grad()

        assert torch.allclose(param1.grad, torch.tensor(0.0))
        assert torch.allclose(param2.grad, torch.tensor(0.0))

    def test_step_skips_parameters_without_gradients(self):
        """Test that step() skips parameters where grad is None"""
        param1 = torch.nn.Parameter(torch.tensor([1.0]))
        param2 = torch.nn.Parameter(torch.tensor([2.0]))
        param1.grad = torch.tensor([0.5])  # Has gradient
        param2.grad = None  # No gradient

        optimizer = SGD([param1, param2], lr=0.1)

        initial_param1 = param1.data.clone()
        initial_param2 = param2.data.clone()

        optimizer.step()

        # param1 should be updated (has grad)
        assert not torch.allclose(param1.data, initial_param1)
        # param2 should remain unchanged (no grad)
        assert torch.allclose(param2.data, initial_param2)

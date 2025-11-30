from typing import Any


class SGD:
    # TODO: fix typing
    def __init__(self, model_parameters: Any, lr: float, lr_decay: float = 1.01):
        self.model_parameters = list(model_parameters)  # list to not exhaust generator
        self.lr = lr
        self.lr_decay = lr_decay

    def step(self):
        """Step function of optimizer to update weights"""
        for param in self.model_parameters:
            if param.grad is not None:
                param.data.sub_(self.lr * param.grad)
        self.lr /= self.lr_decay

    def zero_grad(self):
        """Reset of gradients"""
        for param in self.model_parameters:
            if param.grad is not None:
                param.grad.zero_()

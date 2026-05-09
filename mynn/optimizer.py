from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}
    
    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable == True:
                if layer_idx not in self.velocity:
                    self.velocity[layer_idx] = {
                        key: np.zeros_like(value)
                        for key, value in layer.params.items()
                    }

                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.velocity[layer_idx][key] = self.mu * self.velocity[layer_idx][key] + layer.grads[key]
                    layer.params[key] -= self.init_lr * self.velocity[layer_idx][key]


class RMSProp(Optimizer):
    def __init__(self, init_lr, model, beta=0.9, eps=1e-8):
        super().__init__(init_lr, model)
        self.beta = beta
        self.eps = eps
        self.square_avg = {}

    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable == True:
                if layer_idx not in self.square_avg:
                    self.square_avg[layer_idx] = {
                        key: np.zeros_like(value)
                        for key, value in layer.params.items()
                    }

                for key in layer.params.keys():
                    grad = layer.grads[key]
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    self.square_avg[layer_idx][key] = (
                        self.beta * self.square_avg[layer_idx][key]
                        + (1 - self.beta) * grad * grad
                    )
                    layer.params[key] -= self.init_lr * grad / (np.sqrt(self.square_avg[layer_idx][key]) + self.eps)


class Adam(Optimizer):
    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable == True:
                if layer_idx not in self.m:
                    self.m[layer_idx] = {
                        key: np.zeros_like(value)
                        for key, value in layer.params.items()
                    }
                    self.v[layer_idx] = {
                        key: np.zeros_like(value)
                        for key, value in layer.params.items()
                    }

                for key in layer.params.keys():
                    grad = layer.grads[key]
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)

                    self.m[layer_idx][key] = self.beta1 * self.m[layer_idx][key] + (1 - self.beta1) * grad
                    self.v[layer_idx][key] = self.beta2 * self.v[layer_idx][key] + (1 - self.beta2) * grad * grad
                    m_hat = self.m[layer_idx][key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[layer_idx][key] / (1 - self.beta2 ** self.t)
                    layer.params[key] -= self.init_lr * m_hat / (np.sqrt(v_hat) + self.eps)

import numpy as np

from .tensor import Tensor


class Node:
    BAIS_INIT = 0.0

    def __init__(self, n_inputs, activation=None):
        self.w = [Tensor(np.random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Tensor(self.BAIS_INIT)
        self.activation = activation

    def __call__(self, x):
        _act = (wi * xi for wi, xi in zip(self.w, x))
        act = sum(_act, self.b)
        if self.activation:
            activation_fxn = getattr(act, self.activation)
            act = activation_fxn()
        return act

    def params(self):
        return self.w + [self.b]


class Dense:
    def __init__(self, n_inputs, n_out, name=None, **kwargs):
        self.nodes = [Node(n_inputs, **kwargs) for _ in range(n_out)]
        self.name = name
        self.n_inputs = n_inputs
        self.n_out = n_out

    def __call__(self, x):
        out = [n(x) for n in self.nodes]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.nodes for p in n.params()]

    def summary(self):
        return {
            "name": self.name,
            "shape": (self.n_inputs, self.n_out),
            "params": len(self.parameters()),
        }

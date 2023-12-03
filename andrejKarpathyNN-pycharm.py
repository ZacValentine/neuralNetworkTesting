import torch
from Value import Value
import random
import math
import numpy

# x1 = torch.Tensor([2.0]).double()               ; x1.requires_grad = True
# x2 = torch.Tensor([0.0]).double()               ; x2.requires_grad = True
# w1 = torch.Tensor([-3.0]).double()              ; w1.requires_grad = True
# w2 = torch.Tensor([1.0]).double()               ; w2.requires_grad = True
# b = torch.Tensor([6.8813735870195432]).double() ; x1.requires_grad = True
# n = x1*w1 + x2*w2 + b
# o = torch.tanh(n)
#
# print(o.data.item())
# o.backward()
# print('---')
# print('x2', x2.grad.item())
# print('w2', w2.grad.item())
# print('x1', x1.grad.item())
# print('w1', w1.grad.item())


class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range (len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

n = MLP(3, [4, 4, 1])
xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets # ygt - y ground truth
ypred = [n(x) for x in xs]
print(ypred)

loss = Value(sum((yout.data - ygt)**2 for ygt, yout in zip(ys, ypred))) # should not have .data or Value()
print(loss)

loss.backward()

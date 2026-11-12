# Micrograd

Micrograd is a small, educational automatic-differentiation engine for scalar
values. It builds a computation graph dynamically and applies backpropagation
to calculate gradients. The repository also includes basic neural-network
building blocks constructed on top of the engine.

The implementation is intentionally compact and uses only the Python standard
library, making it useful for learning how autograd and multilayer perceptrons
work under the hood.

## Features

- Scalar-valued reverse-mode automatic differentiation
- Arithmetic operations, powers, division, `tanh`, and ReLU
- Automatic computation-graph traversal during backpropagation
- Neuron, layer, and multilayer perceptron (MLP) abstractions
- Parameter collection and gradient-reset helpers
- No third-party dependencies

## Project structure

```text
micrograd/
|-- engine.py   # Value class and automatic differentiation
|-- nn.py       # Module, Neuron, Layer, and MLP classes
|-- train_mlp.py # Ready-to-run MLP training example
`-- README.md
```

## Requirements

- Python 3.8 or newer

## Run the training example

From the project directory, run:

```bash
python train_mlp.py
```

The script creates an MLP, trains it on a small classification dataset, shows
the loss during training, and prints its final predictions. You can edit the
dataset, network shape, epoch count, and learning rate in `train_mlp.py`.

## Getting started

Clone the repository into a directory named `micrograd`. Run examples from its
parent directory so Python can resolve the package import:

```bash
git clone <repository-url> micrograd
cd <parent-directory>
```

### Automatic differentiation

```python
from micrograd.engine import Value

x = Value(2.0)
y = Value(-3.0)
z = (x * y + x**2).tanh()

z.backward()

print(z)       # output value and its gradient
print(x.grad)  # dz/dx
print(y.grad)  # dz/dy
```

Every operation creates a new `Value` and records its inputs. Calling
`backward()` on the final value walks the graph in reverse topological order
and accumulates gradients in all values that contributed to the result.

### Building and training an MLP

```python
from micrograd.nn import MLP

# Three inputs, two hidden layers of four neurons, and one output.
model = MLP(3, [4, 4, 1])

inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
targets = [1.0, -1.0, -1.0, 1.0]

for step in range(100):
    predictions = [model(x) for x in inputs]
    loss = sum((prediction - target) ** 2
               for prediction, target in zip(predictions, targets))

    model.zero_grad()
    loss.backward()

    learning_rate = 0.05
    for parameter in model.parameters():
        parameter.data -= learning_rate * parameter.grad

print([model(x).data for x in inputs])
```

## Core API

### `Value`

`Value(data)` wraps a scalar and tracks its gradient in `grad`. Supported
operations include addition, subtraction, multiplication, division, numeric
powers, negation, `relu()`, and `tanh()`.

### `Module`

The base neural-network module. `parameters()` returns trainable values and
`zero_grad()` resets their gradients before a new backward pass.

### `Neuron`, `Layer`, and `MLP`

- `Neuron(nin, nonlin=True)` creates a neuron with `nin` inputs.
- `Layer(nin, nout)` creates a layer containing `nout` neurons.
- `MLP(nin, nouts)` stacks layers, where `nouts` contains each layer's size.
  Hidden layers use ReLU and the final layer is linear.

## Notes

This project is designed for learning rather than production workloads. Its
values are scalar, and it does not include tensor operations, GPU acceleration,
serialization, batching, or numerical-stability features found in full deep
learning frameworks.

## License

No license file is currently included. Add one before redistributing or using
the project under explicit licensing terms.

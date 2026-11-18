
import math


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def leaky_relu(self, negative_slope=0.01):
        """Apply a leaky ReLU activation.

        ``negative_slope`` controls the gradient and output for negative inputs.
        """
        if not isinstance(negative_slope, (int, float)):
            raise TypeError("negative_slope must be an int or float")

        out = Value(
            self.data if self.data >= 0 else negative_slope * self.data,
            (self,),
            'LeakyReLU',
        )

        def _backward():
            slope = 1 if self.data >= 0 else negative_slope
            self.grad += slope * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        # math.tanh remains stable for inputs where exp(2*x) would overflow.
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        """Apply a numerically stable logistic sigmoid activation."""
        if self.data >= 0:
            z = math.exp(-self.data)
            s = 1 / (1 + z)
        else:
            z = math.exp(self.data)
            s = z / (1 + z)
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        """Return the natural logarithm (the input must be positive)."""
        if self.data <= 0:
            raise ValueError("log is only defined for positive values")
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward

        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), 'sin')

        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward

        return out

    def cos(self):
        out = Value(math.cos(self.data), (self,), 'cos')

        def _backward():
            self.grad -= math.sin(self.data) * out.grad
        out._backward = _backward

        return out

    def backward(self, gradient=1.0):
        """Backpropagate from this value using ``gradient`` as the seed.

        A custom seed is useful for vector-Jacobian products or when this value
        is part of a larger calculation outside the graph.
        """
        if not isinstance(gradient, (int, float)):
            raise TypeError("gradient must be an int or float")

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = gradient
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __abs__(self):
        """Return the absolute value, using a zero subgradient at zero."""
        out = Value(abs(self.data), (self,), 'abs')

        def _backward():
            sign = 1 if self.data > 0 else -1 if self.data < 0 else 0
            self.grad += sign * out.grad
        out._backward = _backward

        return out

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

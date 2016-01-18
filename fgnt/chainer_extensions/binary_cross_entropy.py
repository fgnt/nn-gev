import numpy

from chainer import cuda
from chainer import function
from chainer.utils import force_array


class BinaryCrossEntropy(function.Function):
    """Binary cross entropy loss."""

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        x, t = inputs
        xp = cuda.get_array_module(x)
        loss = -xp.mean(t * xp.log2(x + 1e-6)
                        + (1 - t) * xp.log2((1 - x) + 1e-6))
        return force_array(loss),

    def backward(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        xp = cuda.get_array_module(x)
        gx = t / (x + 1e-6) - (1 - t) / (1 - x + 1e-6)
        gx *= -gloss / (x.size * xp.log(2).astype(numpy.float32))
        return gx, None


def binary_cross_entropy(x, t):
    """Computes binary entropy loss.

    The values of the input and the target are assumed to be in the range
    between zero and one.

    Args:
        x (Variable): Input matrix
        t (Variable): Target matrix

    Returns:
        Variable: A variable object holding a scalar array of the binary
        cross entropy loss.

    .. note::

       This function is differentiable only by ``x``.

    """
    return BinaryCrossEntropy()(x, t)

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.available:
    pass


class SequenceBatchNormalizationFunction(function.Function):
    """Batch normalization on sequential output.

    This batch normalization is suited for use cases where the dimension of the
    data is `time` x `batch` x `features`. This is often the case in audio
    processing with recurrent neural networks.

    .. note:: Unlike the description in the paper, we don't use a sliding
        window here. We normalize per (batched) utterance(s). This means that
        for decoding, the utterances composing a batch should not change in order
        to keep the result reproducible.

    Args:
        size (int): Size of the features
        eps (float): Epsilon value for numerical stability.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <http://arxiv.org/abs/1502.03167>`_

    .. admonition:: LSTM Example

        Normalization of the input to a LSTM layer.
        Assuming ``x`` is a time series signal, we have no prior information
        of cell content / output, the feature vectors have size ``F`` and the
        batch size is ``B``.
        Most typical preparation is:`

        >>> model = FunctionSet(l_x=SequenceLinear(F, 4*n_units),
                                l_norm=SequenceBatchNormalization(4*n_units),
                                lstm=SequenceLSTM(n_units),
        ...                     ...)
        >>> act = model.l_x(x)
        >>> act_norm = model.l_norm(act)
        >>> y, c, h = model.lstm(act_norm)

    """
    parameter_names = ('gamma', 'beta')
    gradient_names = ('ggamma', 'gbeta')

    def __init__(self, eps=1e-8):
        self.eps = eps

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        x_type, gamma_type, beta_type = in_types

        self_ = type_check.Variable(self, 'self')
        type_check.expect(
                x_type.dtype == numpy.float32,
                x_type.ndim == 3,
                x_type.shape[2] == gamma_type.shape[0],
                x_type.shape[2] == beta_type.shape[0],
        )

    def check_type_backward(self, in_types, out_types):
        pass

    def forward_cpu(self, inputs):
        x, gamma, beta = inputs

        axis = (0, 1)
        xp = cuda.get_array_module(x)

        mean = x.mean(axis=axis, keepdims=True)
        var = x.var(axis=axis, keepdims=True) + self.eps

        self.std = xp.sqrt(var)
        x_mu = x - mean
        self.x_hat = x_mu / self.std
        y = gamma * self.x_hat + beta

        return y,

    def backward(self, inputs, gy):
        x, gamma, beta = inputs
        gy = gy[0]

        axis = (0, 1)
        m = gy.shape[0] * gy.shape[1]

        gbeta = gy.sum(axis=axis)
        ggamma = (gy * self.x_hat).sum(axis=axis)

        coeff = gamma / self.std

        gx = coeff * (gy - self.x_hat * ggamma / m - gbeta / m)
        return gx, ggamma, gbeta

    def forward_gpu(self, inputs):
        x, gamma, beta = inputs

        mean = x.mean(axis=(0, 1), keepdims=True)
        var = x.var(axis=(0, 1), keepdims=True) + self.eps

        normalize = cuda.elementwise(
                'T x, T var, T mean, T gamma, T beta',
                'T std, T x_hat, T y',
                'std = sqrtf(var);'
                'x_hat = (x - mean) / std;'
                'y = gamma * x_hat + beta;',
                'normalize')

        self.std, self.x_hat, y = normalize(x, var, mean, gamma, beta)

        return y,


def sequence_batch_normalization_function(x, gamma, beta):
    return SequenceBatchNormalizationFunction()(x, gamma, beta)

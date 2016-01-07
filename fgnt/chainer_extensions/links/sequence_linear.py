import numpy

from fgnt.chainer_extensions.sequence_linear import sequence_linear_function
from chainer import link
from fgnt.chainer_extensions import weight_init
from fgnt.chainer_extensions.sequenze_batch_normalization import\
    sequence_batch_normalization_function
from chainer.functions.noise.dropout import dropout


class SequenceLinear(link.Link):

    """Sequence linear layer (fully-connected layer/affine transformation).

    This link holds a weight matrix ``W`` and optional a bias vector ``b``.

    The weight matrix ``W`` has shape ``(in_size, out_size)``.
    This matrix is initialized with random uniform distributed values. With a
    scale according to Saxe et al.

    The bias vector ``b`` is of size ``out_size``.
    Each element is initialized with the ``bias`` value.
    If ``nobias`` argument is set to True, then this function does not hold a
    bias vector.

    Let :math:`X` be an input matrix, and :math:`W, b` the weight matrix and
    the bias vector, respectively.
    Then, the output matrix :math:`Y` is computed by :math:`Y = XW + b`,
    where the addition by :math:`b` is broadcasted across the minibatch.

    .. note:: This is the sequential version. Meaning it takes an input of the
        form TxBxF. Before the transformation, this 3D tensor is reshaped to a
        2D matrix with T*BxF so the transformation is applied to each feature
        vector. Afterwards, the matrix is reshaped to the original size again.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimension of output vectors.
        bias (float): Initial bias value.
        nobias (bool): If True, then this function does not use the bias.
        initialW (2-D array): Initial weight value.
        initial_bias (1-D array): Initial bias value.

    """

    def __init__(self, in_size, out_size, bias=0, nobias=False,
                 initialW=None, initial_bias=None, normalized=False):
        super(SequenceLinear, self).__init__(W=(in_size, out_size))
        if initialW is None:
            initialW = weight_init.uniform((in_size, out_size))
        self.W.data[...] = initialW

        if normalized:
            self.add_param('gamma', (out_size,))
            self.add_param('beta', (out_size,))
            self.gamma.data[...] = numpy.ones((out_size,), dtype=numpy.float32)
            self.beta.data[...] = numpy.zeros((out_size,), dtype=numpy.float32)
            nobias = True

        if nobias:
            self.b = None
        else:
            self.add_param('b', out_size)
            if initial_bias is None:
                initial_bias = bias
            self.b.data[...] = initial_bias

        self.add_persistent('normalized', normalized)

    def __call__(self, x, **kwargs):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Time-Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """

        dropout_rate = kwargs.get('dropout', 0.)
        x = dropout(x, dropout_rate)
        x = sequence_linear_function(x, self.W, self.b)
        if self.normalized:
            x = sequence_batch_normalization_function(x, self.gamma, self.beta)
        return x

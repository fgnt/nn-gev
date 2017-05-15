import numpy
from chainer import link

from fgnt.chainer_extensions.sequenze_batch_normalization import \
    sequence_batch_normalization_function


class SequenceBatchNorm(link.Link):
    def __init__(self, size):
        super().__init__()
        self.add_param('gamma', (size,))
        self.add_param('beta', (size,))
        self.gamma.data[...] = numpy.ones((size,), dtype=numpy.float32)
        self.beta.data[...] = numpy.zeros((size,), dtype=numpy.float32)

    def __call__(self, x, **kwargs):
        """Applies the BN layer.

        Args:
            x (~chainer.Variable): Time-Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """

        return sequence_batch_normalization_function(x, self.gamma, self.beta)

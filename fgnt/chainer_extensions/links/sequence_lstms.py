import numpy
from chainer import link
from chainer.functions.array.concat import concat
from chainer.functions.noise.dropout import dropout
from chainer.utils import weight_init

from fgnt.chainer_extensions.sequence_linear \
    import sequence_linear_function
from fgnt.chainer_extensions.sequence_lstm import sequence_lstm_function
from fgnt.chainer_extensions.sequenze_batch_normalization import \
    sequence_batch_normalization_function


class SequenceLSTM(link.Link):
    def __init__(self, in_size, out_size, bias=0, nobias=False,
                 W_x=None, W_h=None, initial_bias=None, reverse=False,
                 normalized=False, stateful=False):
        super(SequenceLSTM, self).__init__(W_x=(in_size, 4 * out_size),
                                           W_h=(out_size, 4 * out_size))
        if W_x is None:
            W_x = weight_init.uniform((in_size, 4 * out_size), low=-0.04,
                                      high=0.04)
        if W_h is None:
            W_h = weight_init.uniform((out_size, 4 * out_size), low=-0.04,
                                      high=0.04)
        self.W_x.data[...] = W_x
        self.W_h.data[...] = W_h

        if normalized:
            self.add_param('gamma', (4 * out_size,))
            self.add_param('beta', (4 * out_size,))
            self.gamma.data[...] = numpy.ones((4 * out_size,),
                                              dtype=numpy.float32)
            self.beta.data[...] = numpy.zeros((4 * out_size,),
                                              dtype=numpy.float32)
            nobias = True

        if nobias:
            self.b = None
        else:
            self.add_param('b', 4 * out_size)
            if initial_bias is None:
                initial_bias = bias
            self.b.data[...] = initial_bias

        self.add_persistent('reverse', reverse)
        self.add_persistent('normalized', normalized)
        self.add_persistent('stateful', stateful)

        self.reset_states()

    def reset_states(self):
        self.h = None
        self.c = None

    def __call__(self, x, **kwargs):
        """Applies the lstm layer.

        Args:
            x (~chainer.Variable): Time-Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the lstm layer.

        """

        dropout_rate = kwargs.get('dropout', 0.)
        dropout_rate_hidden_hidden = kwargs.get('dropout_hidden_hidden', 0.)
        x = dropout(x, dropout_rate)
        lstm_in = sequence_linear_function(x, self.W_x)
        if self.normalized:
            lstm_in = sequence_batch_normalization_function(lstm_in, self.gamma,
                                                            self.beta)
        lstm_out, self.h_prev, self.c_prev = \
            sequence_lstm_function(lstm_in, self.W_h, None, None, self.reverse,
                                   dropout_rate_hidden_hidden)
        return lstm_out


class SequenceBLSTM(link.Chain):
    def __init__(self, in_size, out_size, bias=0, nobias=False,
                 W_x_fw=None, W_h_fw=None, W_x_bw=None, W_h_bw=None,
                 initial_bias=None, normalized=False, concat=False,
                 stateful=False):
        super(SequenceBLSTM, self).__init__(
                lstm_fw=SequenceLSTM(in_size, out_size, bias, nobias,
                                     W_x_fw, W_h_fw, initial_bias,
                                     normalized=normalized,
                                     stateful=stateful),
                lstm_bw=SequenceLSTM(in_size, out_size, bias, nobias,
                                     W_x_bw, W_h_bw, initial_bias,
                                     reverse=True,
                                     normalized=normalized,
                                     stateful=stateful)
        )
        self.add_persistent('concat', concat)
        self.add_persistent('stateful', stateful)

    def __call__(self, x, **kwargs):
        lstm_fw = self.lstm_fw(x, **kwargs)
        lstm_bw = self.lstm_bw(x, **kwargs)
        if self.concat:
            return concat([lstm_fw, lstm_bw], axis=2)
        else:
            return lstm_fw + lstm_bw

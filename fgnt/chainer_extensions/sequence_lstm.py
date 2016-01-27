import chainer
import numpy
import scipy.special
import six
from chainer import cuda
from chainer import function

F32 = numpy.float32


def _extract_gates(x):
    r = x.reshape((x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
    return (r[:, :, i] for i in six.moves.range(4))


def _sigmoid(x):
    return scipy.special.expit(x)


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


_preamble = '''
template <typename T> __device__ T sigmoid(T x) { return 1 / (1 + exp(-x)); }
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    int I = (i+offset) * 4; \
    T aa = tanh(act[I]); \
    T ai = sigmoid(act[I+1]); \
    T af = sigmoid(act[I+2]); \
    T ao = sigmoid(act[I+3]);
'''


class SequenceLSTMFunction(function.Function):
    def __init__(self, reverse=False, dropout=0., dropout_scale=True):
        self.reverse = reverse
        self.dropout_ratio = dropout
        self.dropout_scale = dropout_scale
        self.mask = None

    def check_type_forward(self, in_types):
        pass

    def check_type_backward(self, in_types, out_types):
        pass

    def _flatten(self, val):
        return val.reshape(self.T * self.B, val.shape[2])

    def _deflatten(self, val):
        return val.reshape(self.T, self.B, val.shape[1])

    def forward_cpu(self, inputs):
        x = inputs[0]
        W_h = inputs[1]
        self.T = x.shape[0]
        self.B = x.shape[1]
        self.F = x.shape[2]
        self.units = W_h.shape[0]

        c_prev = inputs[2]
        h_prev = inputs[3]

        if self.mask is None:
            if self.dropout_scale:
                scale = h_prev[0].dtype.type(1. / (1 - self.dropout_ratio))
            else:
                scale = h_prev[0].dtype.type(1.)
            flag = numpy.random.rand(*h_prev.shape) >= self.dropout_ratio
            self.mask = scale * flag

        if self.reverse:
            x = x[::-1]

        self.act = numpy.empty_like(x)
        numpy.copyto(self.act, x)

        self.c_prev = numpy.empty((self.T + 1, self.B, self.units),
                                  dtype=F32)
        self.h_prev = numpy.empty((self.T + 1, self.B, self.units),
                                  dtype=F32)

        self.c_prev[0] = c_prev
        self.h_prev[0] = h_prev

        for t in six.moves.range(self.T):
            self.act[t] += numpy.dot(self.mask * self.h_prev[t], W_h)

            a, i, f, o = _extract_gates(self.act[t])
            a = numpy.tanh(a)
            i = _sigmoid(i)
            f = _sigmoid(f)
            o = _sigmoid(o)

            self.c_prev[t + 1] = a * i + f * self.c_prev[t]
            self.h_prev[t + 1] = o * numpy.tanh(self.c_prev[t + 1])

        if self.reverse:
            return self.h_prev[1:][::-1], self.c_prev[t + 1], self.h_prev[-1]
        else:
            return self.h_prev[1:], self.c_prev[t + 1], self.h_prev[-1]

    def backward_cpu(self, inputs, grad_outputs):
        W_h = inputs[1]
        gout, gc, gh = grad_outputs
        if gc is None:
            gc = numpy.zeros((self.B, self.units), dtype=F32)

        gact = numpy.empty((self.T + 1, self.B, 4 * self.units),
                           dtype=F32)
        gact[-1].fill(0)

        if self.reverse:
            gout = gout[::-1]

        if gh is not None:
            gout = gout.copy()
            gout[-1] += gh

        for t in six.moves.range(self.T - 1, -1, -1):
            a, i, f, o = _extract_gates(self.act[t])
            a = numpy.tanh(a)
            i = _sigmoid(i)
            f = _sigmoid(f)
            o = _sigmoid(o)

            ga, gi, gf, go = _extract_gates(gact[t])

            gh = gout[t] + self.mask * numpy.dot(gact[t + 1], W_h.T)
            co = numpy.tanh(self.c_prev[t + 1])
            gc = (o * gh * _grad_tanh(co) + gc)
            ga[:] = gc * i * _grad_tanh(a)
            gi[:] = gc * a * _grad_sigmoid(i)
            gf[:] = gc * self.c_prev[t] * _grad_sigmoid(f)
            go[:] = gh * co * _grad_sigmoid(o)
            gc *= f  # multiply f here

        gx = gact[:-1]

        gW_h = numpy.dot(self._flatten(self.h_prev[:-1]).T,
                         self._flatten(gx))

        if self.reverse:
            return gx[::-1], gW_h, gc, numpy.dot(gact[0], W_h.T)
        else:
            return gx, gW_h, gc, numpy.dot(gact[0], W_h.T)

    def forward_gpu(self, inputs):
        x = inputs[0]
        W_h = inputs[1]
        self.T = x.shape[0]
        self.B = x.shape[1]
        self.F = x.shape[2]
        self.units = W_h.shape[0]

        c_prev = inputs[2]
        h_prev = inputs[3]

        xp = cuda.cupy

        if self.mask is None:
            if self.dropout_ratio > 0:
                if self.dropout_scale:
                    scale = h_prev[0].dtype.type(1. / (1 - self.dropout_ratio))
                else:
                    scale = h_prev[0].dtype.type(1.)
                flag = xp.random.rand(*h_prev.shape, dtype=numpy.float32) \
                       >= self.dropout_ratio
                self.mask = scale * flag
            else:
                self.mask = xp.ones_like(h_prev)

        if self.reverse:
            self.act = x[::-1]
        else:
            self.act = x.copy()

        self.h_prev = xp.empty((self.T + 1, self.B, self.units), dtype=F32)
        self.c_prev = xp.empty((self.T + 1, self.B, self.units), dtype=F32)

        # Set the hidden and cell for the previous step to zero
        self.c_prev[0] = c_prev
        self.h_prev[0] = h_prev

        lstm_step = cuda.elementwise(
                'raw T act, int64 offset, int64 s',
                'raw T c_prev, raw T h',
                '''
                    COMMON_ROUTINE;
                    c_prev[i + offset + s] = aa * ai + af * c_prev[i+offset];
                    h[i + offset + s] = ao * tanh(c_prev[i + offset + s]);
                ''',
                'lstm_fwd', preamble=_preamble)

        step = self.B * self.units
        for t in six.moves.range(self.T):
            # Add the hidden-hidden activation to the activation calculated
            # outside of the loop
            self.act[t] += (self.mask * self.h_prev[t]).dot(W_h)
            # Apply the LSTM activation
            lstm_step(self.act, t * step, step, self.c_prev, self.h_prev,
                      size=step)

        if self.reverse:
            return self.h_prev[1:][::-1], self.c_prev[-1], \
                   self.h_prev[-1]
        else:
            return self.h_prev[1:], self.c_prev[-1], self.h_prev[-1]

    def backward_gpu(self, inputs, grad_outputs):
        x = inputs[0]
        W_h = inputs[1]
        gout, gc, gh = grad_outputs

        if self.reverse:
            gout = gout[::-1]

        gact = cuda.cupy.empty((self.T, self.B, 4 * self.units), dtype=F32)
        if gc is None:
            gc = cuda.cupy.zeros((self.B, self.units), dtype=F32)
        if gh is None:
            gh = cuda.cupy.zeros((self.B, self.units), dtype=F32)

        # Prepare LSTM kernel
        lstm_grad_step = cuda.elementwise(
                'raw T act, raw T c_prev, raw T gh, raw T gout, raw T mask,'
                'int64 offset, int64 s',
                'raw T gact, raw T gc',
                '''
                   COMMON_ROUTINE;
                   // I = (i+offset) * 4
                   int J = i + offset;
                   T co  = tanh(c_prev[J + s]);
                   T gc1 = (mask[i]*gh[i]+gout[J]) * ao * grad_tanh(co) + gc[i];
                   gact[I+3] = (mask[i]*gh[i]+gout[J]) * co * grad_sigmoid(ao);

                   gc[i]  = gc1 * af;
                   gact[I]         = gc1 * ai        * grad_tanh(aa);
                   gact[I+1]         = gc1 * aa        * grad_sigmoid(ai);
                   gact[I+2]         = gc1 * c_prev[J] * grad_sigmoid(af);
                ''',
                'lstm_bwd', preamble=_preamble)
        step = self.B * self.units

        # Initial timestep to avoid if in loop for gh
        t = self.T - 1
        lstm_grad_step(self.act, self.c_prev, gh, gout, self.mask,
                       t * step, step, gact, gc, size=step)

        for t in six.moves.range(self.T - 2, -1, -1):
            gh = (gact[t + 1]).dot(W_h.T)
            lstm_grad_step(self.act, self.c_prev, gh, gout, self.mask,
                           t * step, step, gact, gc, size=step)

        gW_h = cuda.cupy.dot(self._flatten(self.h_prev[:-1]).T,
                             self._flatten(gact))

        if self.reverse:
            return gact[::-1], gW_h, gc, \
                   cuda.cupy.dot(gact[0], W_h.T)
        else:
            return gact, gW_h, gc, cuda.cupy.dot(gact[0], W_h.T)


def _make_initial_state(xp, batch_size, units):
    """ Creates an initial lstm state filled with zeros

    :param batch_size: batch size to be used for the lstm
    :param volatile: see documentation for `~chainer.Variable`
    :return: c, h filled with zeros
    """
    return xp.zeros((batch_size, units), dtype=numpy.float32)


def sequence_lstm_function(x, W_h, c_prev=None, h_prev=None, reverse=False,
                           dropout=0., dropout_scale=True):
    """Long Short-Term Memory units as a sequential function.

    This function implements LSTM units with forget gates for a batch of time
    series. The first dimension of the data is expected to be the time dimension
    , the second the batch dimension and the third the data dimension. The data
    dimension needs to be 4*`units` as it is used as the activation.
    :math:`\mathbf{W}_h` is the weight matrix for the hidden-hidden
    transformations.
    The advantage of having a single implementation instead of stacking the
    lstm function and linear transformations is the gain in execution speed.
    It is for example possible to move most computation out of the loop over
    the time, resulting in much faster operations on big matrices.

    The forward step expects either a tuple with three elements:
        #. The transformed time series input :math:`x`
        #. The state for the cells
        #. The previous hidden output

    or a single value :math:`x`. In the latter case, the states and the
    previous hidden output are initialized with a zero vector. This should be
    used for full BPTT. For truncated BPTT on the other hand, we need to pass
    the current state of the cells and the last hidden output, hence use the
    first case described above.

    .. note:: The input is expected to be already transformed by a matrix
    :math:`\mathbf{W}_x`

    The output of the function is likewise a tuple with three elements:
        #. The processed time series
        #. The latest state of the cells
        #. The latest output

    Again, for full BPTT, only the first element is of interest, while for
    truncated BPTT the last two elements should be used as an input for the
    next LSTM step.

    The class has the option `inverse_input`. If set to true, the input is
    reversed prior to the application of the transformation. This allows to
    construct bi-directional LSTMs by combining two modules.

    .. note:: The output is also reversed, thus having the "right" time

    Let the previous cell state be :math:`c_{\\text{prev}}`, the previous
    output be :math:`h_{\\text{prev}}` and the incoming signal be :math:`x`.

    We iterate over the time dimension performing the following operations:

        #. Transform :math:`h_{\\text{prev}}` using :math:`\mathbf{W}_h`
            and add it to the activation matrix :math:`\mathbf{A}`
            at timestep :math:`t`. Note, this activation matrix is the input to
            this function

        #. Split the activation :math:`\mathbf{A}(t)` into four arrays
            :math:`a, i, f, o` of the same shapes along the second axis.

            The splitted input signals are corresponding to:

                - :math:`a` : sources of cell input
                - :math:`i` : sources of input gate
                - :math:`f` : sources of forget gate
                - :math:`o` : sources of output gate

        #. Compute output for time :math:`t` as
            .. math::

                c &= \\tanh(a) \\text{sigmoid}(i)
                   + c_{\\text{prev}} \\text{sigmoid}(f), \\\\
                h &= \\tanh(c) \\text{sigmoid}(o).

    Args:
        x (~chainer.Variable): Variable that holds the transformed time series
        c (~chainer.Variable): Variable that holds the cell content of a
            previous LSTM step. It must have the size `batch_size` x `units`.
            If no previous information is available, pass a matrix with zeros.
        h (~chainer.Variable): Variable that holds the last output of a
            previous LSTM step. It must have the size `batch_size` x `units`.
            If no previous information is available, pass a matrix with zeros.
        reverse (bool): Reverse time
        dropout_scale (bool): Scale units after dropout by 1/(1-p)
        dropout (float): dropout ratio for hidden-hidden connection

    Returns:
        tuple: Three :class:`~chainer.Variable` objects ``y``, ``c`` and ``h``.
            ``y`` is the complete output, ``c`` is the latest cell state and
            ``h`` the latest output.

    See the original paper proposing LSTM with forget gates:
    `Long Short-Term Memory in Recurrent Neural Networks \
    <http://www.felixgers.de/papers/phd.pdf>`_.

    """

    if c_prev is None:
        xp = cuda.get_array_module(x.data)
        c_prev = chainer.Variable(
                _make_initial_state(xp, x.data.shape[1], W_h.data.shape[0]),
                name='c_init',
                volatile='auto')

    if h_prev is None:
        xp = cuda.get_array_module(x.data)
        h_prev = chainer.Variable(
                _make_initial_state(xp, x.data.shape[1], W_h.data.shape[0]),
                name='h_init',
                volatile='auto')

    return SequenceLSTMFunction(reverse, dropout, dropout_scale)(
            x, W_h, c_prev, h_prev)

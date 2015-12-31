import numpy

from chainer import cuda
if cuda.available:
    import cupy
    from cupy.cuda.cublas import sgemm
from chainer import function
from chainer.utils import type_check


def _as_mat(x):
    return x.reshape(x.size // x.shape[-1], x.shape[-1])


class SequenceLinearFunction(function.Function):
    """Linear function (a.k.a. fully-connected layer or affine transformation).

    This function holds a weight matrix ``W`` and a bias vector ``b``.

    The weight matrix ``W`` has shape ``(in_size, out_size)``.
    This matrix is initialized with i.i.d. Gaussian samples, each of which has
    zero mean and deviation :math:`\sqrt{1/\\text{in_size}}`.
    The deviation is scaled by factor ``wscale`` if specified.

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

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() >= 2)
        x_type, W, *_ = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 3,
            x_type.shape[2] == W.shape[0],
        )

    def check_type_backward(self, in_types, out_types):
        pass

    def forward_cpu(self, inputs):
        x = inputs[0]
        W = inputs[1]
        x_2d = _as_mat(x)
        Wx = x_2d.dot(W)
        if len(inputs) > 2:
            b = inputs[2]
            Wx += b
        return Wx.reshape(x.shape[0], x.shape[1], -1),

    def forward_gpu(self, inputs):
        x = inputs[0]
        W = inputs[1]
        # Prepare BLAS call for hidden-hidden activation
        handle = cuda.Device().cublas_handle
        k, m = W.shape
        n, l = x.shape[0] * x.shape[1], x.shape[2]
        lda = max(1, x.shape[-1])
        ldb = max(1, W.strides[0] // W.dtype.itemsize)
        ldc = max(1, m)
        Wx = cupy.empty((x.shape[0], x.shape[1], W.shape[1]),
                        dtype=numpy.float32)
        sgemm(handle, False, False, m, n, k, 1, W.data.ptr, ldb,
                  x.data.ptr, lda, 0, Wx.data.ptr, ldc)
        if len(inputs) > 2:
            b = inputs[2]
            Wx += b
        return Wx,

    def backward(self, inputs, gy):
        x = inputs[0]
        W = inputs[1]
        x_2d = _as_mat(x)
        gy_2d = _as_mat(gy[0])
        gW = x_2d.T.dot(gy_2d)

        if len(inputs) > 2:
            gb = gy_2d.sum(0)
            return gy_2d.dot(W.T).reshape(x.shape), gW, gb
        else:
            return gy_2d.dot(W.T).reshape(x.shape), gW


    def backward_gpu(self, inputs, gy):
        x = inputs[0]
        W = inputs[1]
        # Backprop weight
        gW = cuda.cupy.empty_like(W)
        handle = cuda.Device().cublas_handle
        k, n = gy[0].shape[0] * gy[0].shape[1], W.shape[0]
        m = W.shape[1]
        lda = max(1, x.shape[-1])
        ldb = max(1, gy[0].shape[-1])
        ldc = max(1, m)
        sgemm(handle, False, True, m, n, k, 1, gy[0].data.ptr, ldb,
                  x.data.ptr, lda, 1, gW.data.ptr, ldc)
        # Backprop input
        m, k = W.shape
        n, l = x.shape[0] * x.shape[1], gy[0].shape[2]
        lda = max(1, gy[0].shape[-1])
        ldb = max(1, W.shape[1])
        ldc = max(1, m)
        gx = cuda.cupy.empty_like(x)
        sgemm(handle, True, False, m, n, k, 1, W.data.ptr, ldb,
                  gy[0].data.ptr, lda, 0, gx.data.ptr, ldc)
        # Backprop bias
        if len(inputs) > 2:
            gy_2d = _as_mat(gy[0])
            gb = gy_2d.sum(0)
            return gx, gW, gb
        else:
            return gx, gW


def sequence_linear_function(x, W, b=None):
    if b is None:
        return SequenceLinearFunction()(x, W)
    else:
        return SequenceLinearFunction()(x, W, b)
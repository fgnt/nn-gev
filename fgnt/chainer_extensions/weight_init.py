import numpy
from scipy.linalg import svd

def orthogonal(size, sparsity=-1, scale=1, dtype=numpy.float32):
    sizeX = size
    sizeY = size

    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=dtype)
    for dx in range(sizeX):
        perm = numpy.random.permutation(sizeY)
        new_vals = numpy.random.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals

    u, s, v = svd(values)
    values = u * scale

    return values.astype(dtype)


def normal(size, mean=0, variance=0.01, dtype=numpy.float32):
    values = numpy.random.normal(mean, variance, size)

    return values.astype(dtype)


def uniform(size, low=None, high=None, dtype=numpy.float32):
    if dtype is numpy.float32 or dtype is numpy.float64:
        if low is None:
            low = -numpy.sqrt(6. / sum(size))
        if high is None:
            high = numpy.sqrt(6. / sum(size))
        return numpy.asarray(
            numpy.random.uniform(
                low=low,
                high=high,
                size=size
            ),
            dtype=dtype)
    elif dtype is numpy.complex64:
        return (uniform(size, low=low, high=high, dtype=numpy.float32) +
                1j * uniform(size, low=low, high=high, dtype=numpy.float32)).astype(numpy.complex64)
    elif dtype is numpy.complex128:
        return (uniform(size, low=low, high=high, dtype=numpy.float64) +
                1j * uniform(size, low=low, high=high, dtype=numpy.float64)).astype(numpy.complex128)
    else:
        raise ValueError('Requested dtype not available.')


def eye(size, scale=1, dtype=numpy.float32):
    return scale * numpy.eye(size, dtype=dtype)

import numpy as np
from numpy.linalg import solve
from scipy.linalg import eig
from scipy.linalg import eigh


def get_power_spectral_density_matrix(observation, mask=None, sensor_dim=-2,
                                      source_dim=-2, time_dim=-1):
    """
    Calculates the weighted power spectral density matrix.
    It's also called covariance matrix.

    With the *_dim parameters you can change the sort of the dims of the observation and mask.
    But not every combination is allowed.

    :param observation: Complex observations with shape (..., sensors, frames)
    :param mask: Masks with shape (bins, frames) or (..., sources, frames)
    :param sensor_dim: change sensor diemension index (Default: -2)
    :param source_dim: change source diemension index (Default: -2)
    :param time_dim:  change time diemension index (Default: -1), this index must match for mask and observation
    :return: PSD matrix with shape (..., sensors, sensors) or (..., sources, sensors, sensors)

    Examples:
    >>> F, T, D, K = 51, 31, 6, 2
    >>> X = np.random.randn(F, D, T) + 1j * np.random.randn(F, D, T)
    >>> mask = np.random.randn(F, K, T)
    >>> mask = mask / np.sum(mask, axis=0, keepdims=True)
    >>> get_power_spectral_density_matrix(X, mask=mask).shape
    (51, 2, 6, 6)
    >>> mask = np.random.randn(F, T)
    >>> mask = mask / np.sum(mask, axis=0, keepdims=True)
    >>> get_power_spectral_density_matrix(X, mask=mask).shape
    (51, 6, 6)
    """

    # Select the case you have specified
    #   normalize mask is mask is not None
    #   calc psd with einsum
    if mask is None:
        if time_dim == -1 and sensor_dim == -2:
            psd = np.einsum('...dt,...et->...de', observation,
                            observation.conj())
            psd /= observation.shape[-1]
        elif time_dim == -2 and sensor_dim == -1:
            psd = np.einsum('...td,...te->...de', observation,
                            observation.conj())
            psd /= observation.shape[-2]
        elif time_dim == 0 and sensor_dim == 1:
            psd = np.einsum('td...,te...->de...', observation,
                            observation.conj())
            psd /= observation.shape[0]
        else:
            print('time_dim: ', time_dim)
            print('sensor_dim: ', sensor_dim)
            print('observation.shape: ', observation.shape)
            raise NotImplementedError()
    else:
        mask /= np.maximum(np.sum(mask, axis=time_dim, keepdims=True), 1e-10)

        if mask.ndim + 1 == observation.ndim:
            mask = np.expand_dims(mask, sensor_dim)
            source_dim = None
        else:
            mask = np.rollaxis(mask, source_dim, sensor_dim)

        if time_dim == -1 and sensor_dim == -2:
            psd = np.einsum('...kt,...dt,...et->...kde', mask, observation,
                            observation.conj())
            if source_dim is None:
                psd = np.squeeze(psd, axis=-3)
        elif time_dim == -2 and sensor_dim == -1:
            psd = np.einsum('...tk,...td,...te->...kde', mask, observation,
                            observation.conj())
            if source_dim is None:
                psd = np.squeeze(psd, axis=-3)
        else:
            raise NotImplementedError()
    return psd


def get_pca_vector(target_psd_matrix):
    """
    Returns the beamforming vector of a PCA beamformer.
    :param target_psd_matrix: Target PSD matrix
        with shape (..., sensors, sensors)
    :return: Set of beamforming vectors with shape (..., sensors)
    """
    # Save the shape of target_psd_matrix
    shape = target_psd_matrix.shape

    # Reduce independent dims to 1 independent dim
    target_psd_matrix = np.reshape(target_psd_matrix, (-1,) + shape[-2:])

    # Calculate eigenvals/vecs
    eigenvals, eigenvecs = np.linalg.eigh(target_psd_matrix)
    # Find max eigenvals
    vals = np.argmax(eigenvals, axis=-1)
    # Select eigenvec for max eigenval
    beamforming_vector = np.array(
        [eigenvecs[i, :, vals[i]] for i in range(eigenvals.shape[0])])
    # Reconstruct original shape
    beamforming_vector = np.reshape(beamforming_vector, shape[:-1])

    return beamforming_vector


def get_mvdr_vector(atf_vector, noise_psd_matrix):
    """
    Returns the MVDR beamforming vector.

    :param atf_vector: Acoustic transfer function vector
        with shape (..., bins, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (..., bins, sensors)
    """

    while atf_vector.ndim > noise_psd_matrix.ndim - 1:
        noise_psd_matrix = np.expand_dims(noise_psd_matrix, axis=0)

    # Make sure matrix is hermitian
    noise_psd_matrix = 0.5 * (
    noise_psd_matrix + np.conj(noise_psd_matrix.swapaxes(-1, -2)))

    numerator = solve(noise_psd_matrix, atf_vector)
    denominator = np.einsum('...d,...d->...', atf_vector.conj(), numerator)
    beamforming_vector = numerator / np.expand_dims(denominator, axis=-1)

    return beamforming_vector


def get_gev_vector(target_psd_matrix, noise_psd_matrix):
    """
    Returns the GEV beamforming vector.
    :param target_psd_matrix: Target PSD matrix
        with shape (bins, sensors, sensors)
    :param noise_psd_matrix: Noise PSD matrix
        with shape (bins, sensors, sensors)
    :return: Set of beamforming vectors with shape (bins, sensors)
    """
    bins, sensors, _ = target_psd_matrix.shape
    beamforming_vector = np.empty((bins, sensors), dtype=np.complex)
    for f in range(bins):
        try:
            eigenvals, eigenvecs = eigh(target_psd_matrix[f, :, :],
                                        noise_psd_matrix[f, :, :])
        except np.linalg.LinAlgError:
            eigenvals, eigenvecs = eig(target_psd_matrix[f, :, :],
                                       noise_psd_matrix[f, :, :])
        beamforming_vector[f, :] = eigenvecs[:, np.argmax(eigenvals)]
    return beamforming_vector


def blind_analytic_normalization(vector, noise_psd_matrix):
    bins, sensors = vector.shape
    normalization = np.zeros(bins)
    for f in range(bins):
        normalization[f] = np.abs(np.sqrt(np.dot(
            np.dot(np.dot(vector[f, :].T.conj(), noise_psd_matrix[f]),
                   noise_psd_matrix[f]), vector[f, :])))
        normalization[f] /= np.abs(np.dot(
            np.dot(vector[f, :].T.conj(), noise_psd_matrix[f]), vector[f, :]))

    return vector * normalization[:, np.newaxis]


def apply_beamforming_vector(vector, mix):
    return np.einsum('...a,...at->...t', vector.conj(), mix)


def gev_wrapper_on_masks(mix, noise_mask=None, target_mask=None,
                         normalization=False):
    if noise_mask is None and target_mask is None:
        raise ValueError('At least one mask needs to be present.')

    mix = mix.T
    if noise_mask is not None:
        noise_mask = noise_mask.T
    if target_mask is not None:
        target_mask = target_mask.T

    if target_mask is None:
        target_mask = np.clip(1 - noise_mask, 1e-6, 1)
    if noise_mask is None:
        noise_mask = np.clip(1 - target_mask, 1e-6, 1)

    target_psd_matrix = get_power_spectral_density_matrix(mix, target_mask)
    noise_psd_matrix = get_power_spectral_density_matrix(mix, noise_mask)

    target_psd_matrix -= noise_psd_matrix

    # Beamforming vector
    W_gev = get_gev_vector(target_psd_matrix, noise_psd_matrix)

    if normalization:
        W_gev = blind_analytic_normalization(W_gev, noise_psd_matrix)

    output = apply_beamforming_vector(W_gev, mix)

    return output.T

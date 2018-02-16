import numpy as np
from numpy.linalg import solve
from scipy.linalg import eig
from scipy.linalg import eigh


def get_power_spectral_density_matrix(observation, mask=None, normalize=True):
    """
    Calculates the weighted power spectral density matrix.

    This does not yet work with more than one target mask.

    :param observation: Complex observations with shape (bins, sensors, frames)
    :param mask: Masks with shape (bins, frames) or (bins, 1, frames)
    :return: PSD matrix with shape (bins, sensors, sensors)
    """
    bins, sensors, frames = observation.shape

    if mask is None:
        mask = np.ones((bins, frames))
    if mask.ndim == 2:
        mask = mask[:, np.newaxis, :]

    psd = np.einsum('...dt,...et->...de', mask * observation,
                    observation.conj())
    if normalize:
        normalization = np.sum(mask, axis=-1, keepdims=True)
        psd /= normalization
    return psd


def condition_covariance(x, gamma):
    """see https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3)"""
    scale = gamma * np.trace(x) / x.shape[-1]
    scaled_eye = np.eye(x.shape[-1]) * scale
    return (x + scaled_eye) / (1 + gamma)


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
            beamforming_vector[f, :] = eigenvecs[:, -1]
        except np.linalg.LinAlgError:
            print('LinAlg error for frequency {}'.format(f))
            beamforming_vector[f, :] = (
                np.ones((sensors,)) / np.trace(noise_psd_matrix[f]) * sensors
            )
    return beamforming_vector


def blind_analytic_normalization_legacy(vector, noise_psd_matrix):
    bins, sensors = vector.shape
    normalization = np.zeros(bins)
    for f in range(bins):
        normalization[f] = np.abs(np.sqrt(np.dot(
                np.dot(np.dot(vector[f, :].T.conj(), noise_psd_matrix[f]),
                       noise_psd_matrix[f]), vector[f, :])))
        normalization[f] /= np.abs(np.dot(
                np.dot(vector[f, :].T.conj(), noise_psd_matrix[f]),
                vector[f, :]))

    return vector * normalization[:, np.newaxis]


def blind_analytic_normalization(vector, noise_psd_matrix, eps=0):
    """Reduces distortions in beamformed ouptput.
        
    :param vector: Beamforming vector
        with shape (..., sensors)
    :param noise_psd_matrix:
        with shape (..., sensors, sensors)
    :return: Scaled Deamforming vector
        with shape (..., sensors)
    
    >>> vector = np.random.normal(size=(5, 6)).view(np.complex128)
    >>> vector.shape
    (5, 3)
    >>> noise_psd_matrix = np.random.normal(size=(5, 3, 6)).view(np.complex128)
    >>> noise_psd_matrix = noise_psd_matrix + noise_psd_matrix.swapaxes(-2, -1)
    >>> noise_psd_matrix.shape
    (5, 3, 3)
    >>> w1 = blind_analytic_normalization_legacy(vector, noise_psd_matrix)
    >>> w2 = blind_analytic_normalization(vector, noise_psd_matrix)
    >>> np.testing.assert_allclose(w1, w2)
        
    """
    nominator = np.einsum(
        '...a,...ab,...bc,...c->...',
        vector.conj(), noise_psd_matrix, noise_psd_matrix, vector
    )
    nominator = np.abs(np.sqrt(nominator))

    denominator = np.einsum(
        '...a,...ab,...b->...', vector.conj(), noise_psd_matrix, vector
    )
    denominator = np.abs(denominator)

    normalization = nominator / (denominator + eps)
    return vector * normalization[..., np.newaxis]


def apply_beamforming_vector(vector, mix):
    return np.einsum('...a,...at->...t', vector.conj(), mix)


def phase_correction(vector):
    """Phase correction to reduce distortions due to phase inconsistencies.
    Args:
        vector: Beamforming vector with shape (..., bins, sensors).
    Returns: Phase corrected beamforming vectors. Lengths remain.
    """
    w = vector.copy()
    F, D = w.shape
    for f in range(1, F):
        w[f, :] *= np.exp(-1j*np.angle(
            np.sum(w[f, :] * w[f-1, :].conj(), axis=-1, keepdims=True)))
    return w


def gev_wrapper_on_masks(mix, noise_mask=None, target_mask=None,
                         normalization=False):
    if noise_mask is None and target_mask is None:
        raise ValueError('At least one mask needs to be present.')

    org_dtype = mix.dtype
    mix = mix.astype(np.complex128)
    mix = mix.T
    if noise_mask is not None:
        noise_mask = noise_mask.T
    if target_mask is not None:
        target_mask = target_mask.T

    target_psd_matrix = get_power_spectral_density_matrix(
        mix, target_mask, normalize=False)
    noise_psd_matrix = get_power_spectral_density_matrix(
        mix, noise_mask, normalize=True)
    noise_psd_matrix = condition_covariance(noise_psd_matrix, 1e-6)
    noise_psd_matrix /= np.trace(
        noise_psd_matrix, axis1=-2, axis2=-1)[..., None, None]
    W_gev = get_gev_vector(target_psd_matrix, noise_psd_matrix)
    W_gev = phase_correction(W_gev)

    if normalization:
        W_gev = blind_analytic_normalization(W_gev, noise_psd_matrix)

    output = apply_beamforming_vector(W_gev, mix)
    output = output.astype(org_dtype)

    return output.T

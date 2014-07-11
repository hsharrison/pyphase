"""pyphase.discrete

Algorithms for calculating discrete relative phase.

"""
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA


def relative_phase(a, b=None, t=None):
    """Calculate discrete relative phase between two signals.

    Parameters
    ----------
    a : array-like
    b : array-like
        Periodic signals.
        If only `a` is passed, it must have two columns;
        the columns will be interpreted as the signals.
        If `b` is passed, and the arrays have more than one column,
        they will be collapsed into vectors using principal component analysis.
    t : array-like, optional
        Time array.
        Used for returning cycle times.
        If not passed, cycle indices will be returned.

    Returns
    -------
    t_a : array
    t_b : array
        Time positions of beginnings of cycles in `a` and `b`, where relative phase is estimated.
    rel_phase : array
        Relative phase, in radians.

    Notes
    -----
    Discrete relative phase is based on the assumptions that the two signals have a one-to-one frequency ratio
    and a sinusoidal time history [1]_.

    DRP is calculated according to the equation [1]_

    .. math:: \phi_j = 2 \pi \frac{ t_{\textrm{max }a,j  } - t_{\textrm{max }b,j} }
                                  { t_{\textrm{max }a,j+1} - t_{\textrm{max }a,j} },

    where :math:`t_{\textrm{max }k,i} is the time of the :math:`i`th maximum of array :math:`k`,
    and :math:`\phi_i` is the relative phase for cycle :math:`i`.

    References
    ----------
    .. [1] Wheat, J.S. & Glazier, P.S. (2006).
    Measuring coordination and variability in coordination.
    In Davids, K., Bennett, S., & Newell, K. (eds.), *Movement System Variability* (pp. 167-181).
    Champaign, IL: Human Kinetics.

    """
    a, b = _validate_vectors(a, b)
    if t is None:
        t = np.arange(a.shape[0])

    peaks = [ampd_peaks(a), ampd_peaks(b)]
    periods = np.diff(peaks[0])

    # Note: no effort is made to make sure peaks line up.
    # (This should either be fixed--if a method to do so can be found in the literature--
    # or warned about--if we can specify the effects of having mismatched peaks.)
    # Instead we just crop the vectors to make sure they are the right size
    length = min([peaks[0].shape[0], peaks[1].shape[0], periods.shape[0]])
    peaks[0] = peaks[0][:length]
    peaks[1] = peaks[1][:length]
    periods = periods[:length]

    rel_phase = 2 * np.pi * (peaks[0] - peaks[1]) / periods

    return t[peaks[0]], t[peaks[1]], rel_phase


def ampd_peaks(data):
    """Automatic multiscale-based peak detection.

    A robust peak-detection algorithm suitable for noisy periodic or quasi-periodic data,
    with no free parameters [1]_.

    Parameters
    ----------
    data : array-like
        A periodic or quasi-periodic signal, in a 1D array (or squeezable to 1D).

    Returns
    -------
    peaks : array
        Indices of peaks in `data`.

    References
    ----------
    .. [1] Scholkmann, F., Boss, J., & Wolf, M. (2012).
    An efficient algorithm for automatic peak detection in noisy periodic and quasi-periodic signals.
    *Algorithms, 5*, 588-603.


    """
    # 1. Calculate LMS.
    lms = _local_maxima_scalogram(data)

    # 2. Row-wise sum.
    gamma = lms.sum(axis=1)

    # 3. Remove all rows from lms with k > lambda.
    global_min = gamma.argmin()  # lambda

    # 4. Calculate column-wise standard deviation.
    stds = lms[:global_min + 1].std(axis=0, ddof=1)

    # 5. Return all indices i for which std_i == 0.
    return np.nonzero(stds == 0)[0]


def _local_maxima_scalogram(data):
    a = 1

    data = signal.detrend(_validate_vector(data))
    N = data.shape[0]
    L = int(np.ceil(N/2) - 1)

    # First, construct the LMS as if all the elements were r + a.
    # Then we'll change the other values to zero.
    lms = np.random.sample((L, N)) + a

    # Create shifted forward and shifted back matrices the same size as the LMS.
    shifted_back = np.empty(lms.shape)
    shifted_forward = np.empty(lms.shape)
    # The choice of zeroes seems to be based on the previous element (i -1) so we just roll the data forward 1.
    for shift_ix in range(L):
        shift = shift_ix + 1
        shifted_back[shift_ix, :] = np.roll(data, -shift)
        shifted_forward[shift_ix, :] = np.roll(data, shift)

    # We only want to check elements with k + 1 <= i <= N - k.
    # i: index in original data.
    # k: magnitude of shift.
    shift, data_ix = np.indices(lms.shape)
    shift += 1
    elements_to_check = (shift + 1 <= data_ix) & (data_ix <= N - shift)

    # To do the comparisons in one sweep we need a tiled version of data.
    data = np.tile(data, (L, 1))
    lms[elements_to_check & (data > shifted_back) & (data > shifted_forward)] = 0

    return lms


def _validate_vector(data):
    data = np.squeeze(data)
    if len(data.shape) > 1:
        raise ValueError('input cannot be cast to 1D vector')
    return data


def _validate_vectors(a, b):
    if b is None:
        a = np.squeeze(a)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError('with one input, array must have be 2D with two columns')
        a, b = a[:, 0], a[:, 1]

    a = np.squeeze(a)
    if a.ndim > 1:
        a = _collapse(a)

    b = np.squeeze(b)
    if b.ndim > 1:
        b = _collapse(b)

    return a, b


def _collapse(data):
    return np.squeeze(PCA(n_components=1).fit_transform(data))

import numpy as np


def wrap(phase):
    """Wrap periodic data to fit in the range ``[-pi, pi]``.

    Parameters
    ----------
    phase : array-like
        Any circular data such that ``-pi`` has the same meaning as ``pi``.

    Returns
    -------
    wrapped : array-like
        Wrapped copy of `phase`.

    See Also
    --------
    numpy.unwrap

    """
    return (np.asarray(phase) + np.pi) % (2 * np.pi) - np.pi

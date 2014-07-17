import numpy as np

from pyphase import util


def test_wrap():
    assert np.all(np.isclose(util.wrap([0, 2 * np.pi, -2 * np.pi, np.pi, -np.pi]),
                             [0, 0, 0, -np.pi, -np.pi]))

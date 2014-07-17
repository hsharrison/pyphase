import numpy as np
from scipy import signal
import pytest

from pyphase import discrete


def test_ampd_peaks():
    x = np.sin(np.arange(0, 20*np.pi, np.pi/20))
    noise = np.random.sample(size=x.shape)/100  # test with just a little noise
    actual_peaks = signal.argrelmax(x)[0]
    detected_peaks = discrete.ampd_peaks(x + noise)
    assert set(detected_peaks) - set(actual_peaks) == set()
    # Grant forgiveness for first and last peaks.
    assert set(actual_peaks) - set(detected_peaks) - {actual_peaks[0], actual_peaks[-1]} == set()


def test_ampd_peaks_bad_input():
    with pytest.raises(ValueError):
        discrete.ampd_peaks(np.random.sample((10, 10)))


def test_rel_phase_in_phase():
    x = np.sin(np.arange(0, 20*np.pi, np.pi/20))
    t_1, t_2, rel_phase = discrete.relative_phase(x, 10*x)
    assert np.all(rel_phase == 0)
    assert np.all(t_1 == t_2)


def test_rel_phase_from_two_columns():
    x = np.sin(np.arange(0, 20*np.pi, np.pi/20))
    t_1, t_2, rel_phase = discrete.relative_phase(np.vstack((x, 10*x)).transpose())
    assert np.all(rel_phase == 0)
    assert np.all(t_1 == t_2)


def test_rel_phase_with_time():
    t = np.arange(0, 20*np.pi, np.pi/20)
    t_x, t_y, rel_phase = discrete.relative_phase(np.sin(t), np.cos(t), t)
    assert np.all(np.isclose(t_x - t_y, np.pi/2))


def test_rel_phase_antiphase():
    x = np.sin(np.arange(0, 20*np.pi, np.pi/20))
    y = np.cos(np.arange(0, 20*np.pi, np.pi/20))
    t_x, t_y, rel_phase = discrete.relative_phase(x, y)
    assert np.all(np.isclose(rel_phase, np.pi/2))
    assert np.all(t_x - t_y == 10) or np.all(t_y - t_x == 30)


def test_rel_phase_with_2d_data():
    x = np.sin(np.arange(0, 20*np.pi, np.pi/20))
    y = np.cos(np.arange(0, 20*np.pi, np.pi/20))
    x = np.array([x, x/5]).transpose()
    y = np.array([y, 10*y + 4]).transpose()
    t_x, t_y, rel_phase = discrete.relative_phase(x, y)
    assert np.all(np.isclose(rel_phase, np.pi/2))
    assert np.all(t_x - t_y == 10) or np.all(t_y - t_x == 30)


def test_rel_phase_bad_input():
    with pytest.raises(ValueError):
        discrete.relative_phase(np.arange(100))

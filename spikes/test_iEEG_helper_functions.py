import pytest
import numpy as np
from iEEG_helper_functions import *


def test_basic_selection():
    labels = ["EKG1", "FP1", "C3", "T7", "O2", "O1"]
    expected = np.array([False, False, False, True, False, True])
    assert np.array_equal(electrode_selection(labels), expected)


def test_case_insensitivity():
    labels = ["eKg1", "Fp1", "c3", "T7"]
    expected = np.array([False, False, False, True])
    assert np.array_equal(electrode_selection(labels), expected)


def test_other_labels():
    labels = ["ABC", "XYZ", "RATE1"]
    expected = np.array([True, True, False])
    assert np.array_equal(electrode_selection(labels), expected)


def test_o2_logic():
    labels = ["O2", "O1"]
    expected = np.array([False, True])
    assert np.array_equal(electrode_selection(labels), expected)

    labels = ["O2"]
    expected = np.array([True])
    assert np.array_equal(electrode_selection(labels), expected)


def test_empty_labels():
    labels = []
    expected = np.array([], dtype=bool)
    assert np.array_equal(electrode_selection(labels), expected)


def test_label_set_vs_individual():
    labels = ["T3", "T4", "T5", "T6", "EKG", "ECG"]
    expected = np.array([False, False, False, False, False, False])
    assert np.array_equal(electrode_selection(labels), expected)

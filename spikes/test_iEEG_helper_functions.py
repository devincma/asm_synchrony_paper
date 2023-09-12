import pytest
import numpy as np
from iEEG_helper_functions import *


def test_removes_unwanted_labels():
    labels = ["EKG123", "ECG", "HeartRate", "RRInterval", "Normal"]
    expected = [False, False, False, False, True]
    assert list(electrode_selection(labels)) == expected


def test_removes_check_labels():
    labels = ["C3", "C4", "CZ", "Normal"]
    expected = [False, False, False, True]
    assert list(electrode_selection(labels)) == expected


def test_special_condition_for_O2():
    labels_with_O1 = ["O1", "O2"]
    labels_without_O1 = ["O2"]

    expected_with_O1 = [True, False]
    expected_without_O1 = [True]

    assert list(electrode_selection(labels_with_O1)) == expected_with_O1
    assert list(electrode_selection(labels_without_O1)) == expected_without_O1

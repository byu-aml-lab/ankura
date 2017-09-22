"""Tests for eval"""

import pytest

from ankura.validate import *


def test_bad_xref():
    """Tests that bad xref will get correct precision, recall and f-measure"""
    contingency = Contingency()
    contingency[True, True] = 1 # TP
    contingency[False, True] = 100 # FP
    contingency[True, False] = 10 # FN
    contingency[False, False] = 1000 # TN
    assert contingency[None, None] == 1111
    assert contingency.precision() == 1 / 101
    assert contingency.recall() == 1 / 11
    assert contingency.fmeasure() == 2 * 1/11 * 1/101 / (1/11 + 1/101)


def test_contingency_set_sums():
    """Tests that contingency raises KeyError if setting sums is attempted"""
    contingency = Contingency()
    with pytest.raises(KeyError):
        contingency[None, None] = 1
    with pytest.raises(KeyError):
        contingency[None, True] = 1
    with pytest.raises(KeyError):
        contingency[True, None] = 1

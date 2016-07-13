"""Test functions for iterators module"""

import metis

def test_sequential_pairs():
    """should generate all pairs"""
    pairs = list(metis.iterators.sequential_pairs(range(3)))
    assert pairs == [(0, 1), (1, 2)]

def test_sequential_pairs_offset():
    """should allow an offset"""
    pairs = list(metis.iterators.sequential_pairs(range(5), offset=2))
    assert pairs == [(2, 3), (3, 4)]

def test_circular_pairs():
    """should generate all pairs"""
    pairs = list(metis.iterators.circular_pairs(range(3)))
    assert pairs == [(0, 1), (1, 2), (2, 0)]

def test_circular_pairs_offset():
    """should allow an offset"""
    pairs = list(metis.iterators.circular_pairs(range(5), offset=2))
    assert pairs == [(2, 3), (3, 4), (4, 0), (0, 1), (1, 2)]

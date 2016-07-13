"""Utilities for iteration

This module provides some common utilities for iteration not provided by
the python standard library.
"""

def sequential_pairs(seq, offset=0):
    """Generate all sequential pairs of elements of a sequence

    Args:
        offset (int): if positive, ignore this many pairs before
            returning. Negative values are ignored.

    Yields:
        tuple: Pairs of items from the sequence

    Examples:
        >>> list(sequential_pairs(xrange(5)))
        [(0, 1), (1, 2), (2, 3), (3, 4)]
        >>> list(sequential_pairs(xrange(5), offset=2))
        [(2, 3), (3, 4)]
    """
    i = iter(seq)
    for _ in xrange(offset):
        i.next()
    prev = item = i.next()
    for item in i:
        yield prev, item
        prev = item

def circular_pairs(seq, offset=0):
    """Generate sequential pairs of items from a circular sequence

    Args:
        offset (int):

    Yields:
        tuple: Pairs of items from the sequence, beginning with item
            index,`offset`, including the pair consisting of the last
            and first item

    Examples:
        >>> list(circular_pairs(xrange(5)))
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        >>> list(circular_pairs(xrange(5), offset=2))
        [(2, 3), (3, 4), (4, 0), (0, 1), (1, 2)]
    """
    i = iter(seq)
    for _ in xrange(offset):
        i.next()
    first = prev = item = i.next()
    for item in i:
        yield prev, item
        prev = item

    i = iter(seq)
    while item != first:
        item = i.next()
        yield prev, item
        prev = item

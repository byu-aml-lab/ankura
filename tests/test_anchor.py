"""Tests for anchor"""

import numpy

import pytest

from ankura import pipeline, anchor


@pytest.mark.skip(reason='test implementation missing')
def test_gram_schmidt():
    """Tests gram_schmidt"""


def test_build_cooccurrence1():
    """Tests build_cooccurrence (example 1)"""
    corpus = pipeline.Corpus(
        [
            pipeline.Document(
                'dog dog',
                [pipeline.TokenLoc(0, 0), pipeline.TokenLoc(0, 4)],
                {},
            ),
            pipeline.Document(
                'cat dog',
                [pipeline.TokenLoc(1, 0), pipeline.TokenLoc(0, 4)],
                {},
            ),
        ],
        ['dog', 'cat'],
    )
    expected = numpy.array([[.5, .25],
                            [.25, 0]])
    actual = anchor.build_cooccurrence(corpus)
    assert numpy.allclose(expected, actual)


def test_build_cooccurrence2():
    """Tests build_cooccurrence (example 2)"""
    corpus = pipeline.Corpus(
        [
            pipeline.Document(
                'dog dog',
                [
                    pipeline.TokenLoc(0, 0),
                    pipeline.TokenLoc(0, 4),
                ],
                {},
            ),
            pipeline.Document(
                'cat dog',
                [
                    pipeline.TokenLoc(1, 0),
                    pipeline.TokenLoc(0, 4),
                ],
                {},
            ),
            pipeline.Document(
                'cat cat cat',
                [
                    pipeline.TokenLoc(1, 0),
                    pipeline.TokenLoc(1, 4),
                    pipeline.TokenLoc(1, 8),
                ],
                {},
            ),
            pipeline.Document(
                'cat cat dog',
                [
                    pipeline.TokenLoc(1, 0),
                    pipeline.TokenLoc(1, 4),
                    pipeline.TokenLoc(0, 8),
                ],
                {},
            ),
        ],
        ['dog', 'cat'],
    )
    expected = numpy.array([[6/24, 5/24],
                            [5/24, 8/24]])
    actual = anchor.build_cooccurrence(corpus)
    assert numpy.allclose(expected, actual)


@pytest.mark.skip(reason='test implementation missing')
def test_recover_topics():
    """Tests recover_topics"""


def test_tandem_anchors():
    """Tests tandem anchors"""
    Q = numpy.array([[1, 2, 3, 4, 5, 6],
                     [1, 2, 3, 4, 5, 6],
                     [6, 5, 4, 3, 2, 1],
                     [6, 5, 4, 3, 2, 1],
                     [3, 2, 1, 6, 5, 4],
                     [6, 5, 4, 3, 2, 1]], dtype=float)
    anchors = [[0, 1], [3, 4, 5]]
    expected = numpy.array([[1, 2, 3, 4, 5, 6],
                            [3/(1/6+1/3+1/6),
                             3/(1/5+1/2+1/5),
                             3/(1/4+1/1+1/4),
                             3/(1/3+1/6+1/3),
                             3/(1/2+1/5+1/2),
                             3/(1/1+1/4+1/1)]])
    actual = anchor.tandem_anchors(anchors, Q)
    assert numpy.allclose(expected, actual)


def test_tandem_anchors_vocab():
    """Tests tandem anchors"""
    Q = numpy.array([[1, 2, 3, 4, 5, 6],
                     [1, 2, 3, 4, 5, 6],
                     [6, 5, 4, 3, 2, 1],
                     [6, 5, 4, 3, 2, 1],
                     [3, 2, 1, 6, 5, 4],
                     [6, 5, 4, 3, 2, 1]], dtype=float)
    anchors = [[0, 1], [3, 4, 5]]
    expected = numpy.array([[1, 2, 3, 4, 5, 6],
                            [3/(1/6+1/3+1/6),
                             3/(1/5+1/2+1/5),
                             3/(1/4+1/1+1/4),
                             3/(1/3+1/6+1/3),
                             3/(1/2+1/5+1/2),
                             3/(1/1+1/4+1/1)]])
    actual = anchor.tandem_anchors(anchors, Q)
    assert numpy.allclose(expected, actual)

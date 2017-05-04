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
                [pipeline.TypeLoc(0, 0), pipeline.TypeLoc(0, 4)],
                {},
            ),
            pipeline.Document(
                'cat dog',
                [pipeline.TypeLoc(1, 0), pipeline.TypeLoc(0, 4)],
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
                    pipeline.TypeLoc(0, 0),
                    pipeline.TypeLoc(0, 4),
                ],
                {},
            ),
            pipeline.Document(
                'cat dog',
                [
                    pipeline.TypeLoc(1, 0),
                    pipeline.TypeLoc(0, 4),
                ],
                {},
            ),
            pipeline.Document(
                'cat cat cat',
                [
                    pipeline.TypeLoc(1, 0),
                    pipeline.TypeLoc(1, 4),
                    pipeline.TypeLoc(1, 8),
                ],
                {},
            ),
            pipeline.Document(
                'cat cat dog',
                [
                    pipeline.TypeLoc(1, 0),
                    pipeline.TypeLoc(1, 4),
                    pipeline.TypeLoc(0, 8),
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

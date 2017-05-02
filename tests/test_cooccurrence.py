"""Tests for cooccurrence"""

import numpy

from ankura import pipeline, cooccurrence


def test_build1():
    """Tests build (example 1)"""
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
    actual = cooccurrence.build(corpus)
    assert numpy.allclose(expected, actual)


def test_build2():
    """Tests build (example 2)"""
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
    actual = cooccurrence.build(corpus)
    assert numpy.allclose(expected, actual)

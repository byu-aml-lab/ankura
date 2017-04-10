"""Tests for cooccurrence"""

import numpy

from ankura import pipeline, cooccurrence


def test_build():
    """Tests build"""
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
    assert numpy.array_equal(expected, actual)

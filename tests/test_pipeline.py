"""Tests for ankura.pipeline"""

import io
import ankura


def _mock_file(name, data):
    docfile = io.BytesIO(data.encode())
    docfile.name = name
    return docfile


def test_whole_extractor():
    """Tests ankura.pipeline.whole_extractor"""
    name, data = 'name', 'lorem ipsum\ndolar set'
    docfile = _mock_file(name, data)
    expected = [ankura.pipeline.Text(name, data)]
    actual = list(ankura.pipeline.whole_extractor()(docfile))
    assert expected == actual

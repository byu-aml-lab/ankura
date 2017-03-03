"""Tests for pipeline"""

import gzip
import io
import tarfile

import pytest

# pylint: disable=wildcard-import,unused-wildcard-import
from ankura.pipeline import *


def _mock_file(name, data):
    docfile = io.BytesIO(data.encode())
    docfile.name = name
    return docfile


def test_whole_extractor():
    """Tests whole_extractor"""
    name, data = 'name', 'lorem ipsum\ndolar set'
    docfile = _mock_file(name, data)
    expected = [Text(name, data)]
    actual = list(whole_extractor()(docfile))
    assert expected == actual


def test_skip_extractor():
    """Tests skip_extractor"""
    name, delim = 'name', '\n\n'
    docfile = _mock_file(name, 'header\nheader\n\nlorem ipsum\ndolar set.')
    expected = [Text(name, 'lorem ipsum\ndolar set.')]
    actual = list(skip_extractor(delim)(docfile))
    assert expected == actual


def test_skip_extractor_err():
    """Tests the failure case for skip_extractor"""
    docfile = _mock_file('name', 'header\nheader\n\nlorem ipsum\ndolar set.')
    with pytest.raises(ValueError):
        list(skip_extractor('$$$')(docfile))


def test_line_extractor():
    """Tests line_extractor"""
    docfile = _mock_file('file', ' lorem ipsum dolar set\nasdf qwer zxcv')
    expected = [
        Text('lorem', 'ipsum dolar set'),
        Text('asdf', 'qwer zxcv'),
    ]
    actual = list(line_extractor(' ')(docfile))
    assert expected == actual


def test_line_extractor_err():
    """Tests the failure case for line_extractor"""
    docfile = _mock_file('file', 'lorem$$$ipsum\nasdf qwer\nfoo$$$bar')
    with pytest.raises(ValueError):
        list(line_extractor('$$$')(docfile))


def test_html_extractor():
    """Tests html_extractor"""
    name = 'page.html'
    html = """<!doctype html>
<html>
    <head>
        <title>Test Page</title>
        <meta charset="utf-8" />
    </head>
    <body>
        <h1>Test Page</h1>
        <p>Lorem ipsum dolar set</p>
    </body>
</html>
"""
    output = """Test Page
Test Page
Lorem ipsum dolar set"""
    docfile = _mock_file(name, html)
    expected = [Text(name, output)]
    actual = list(html_extractor()(docfile))
    assert expected == actual


def test_targz_extractor():
    """Tests targz_extractor, which in turn relies on
    ankura.gzip_extractor and ankura.tar_extractor
    """
    expected = [
        Text('lorem.txt', 'lorem ipsum dolar set'),
        Text('asdf.txt', 'asdf asdf asdf'),
        Text('foobar.txt', 'foobar foobaz spam eggs'),
    ]

    tbuf = io.BytesIO()
    tfile = tarfile.TarFile(fileobj=tbuf, mode='w')
    for text in expected:
        data = text.data.encode()
        info = tarfile.TarInfo(text.name)
        info.size = len(data)
        tfile.addfile(info, io.BytesIO(data))
    tfile.close()
    tbuf.seek(0)

    zbuf = io.BytesIO()
    zfile = gzip.GzipFile(fileobj=zbuf, mode='w')
    zfile.write(tbuf.read())
    zfile.close()
    zbuf.seek(0)

    actual = list(targz_extractor(whole_extractor())(zbuf))
    assert actual == expected


def test_default_tokenizer():
    """Tests ankura.default_tokenizer, which in turn relies on
    ankura.translate_tokenizer and ankura.split_tokenizer
    """
    #       0         1           2         3         4
    #       012345678901 23456 78901234567890123 456789
    data = 'Lorem Ipsum\t$100\n<token> asdf don\'t 1230'
    expected = [
        TokenLoc('lorem', 0),
        TokenLoc('ipsum', 6),
        TokenLoc('100', 12),
        TokenLoc('token', 17),
        TokenLoc('asdf', 25),
        TokenLoc('dont', 30),
        TokenLoc('1230', 36),
    ]
    actual = default_tokenizer()(data)
    assert actual == expected


def test_regex_tokenizer():
    """Tests ankura.pipeline.regex_tokenizer"""
    #       0         1         2         3
    #       0123456789012345678901234567890
    data = 'Lorem Ipsum $100 token weird$69'
    expected = [
        TokenLoc('Lorem', 0),
        TokenLoc('Ipsum', 6),
        TokenLoc('<money>', 12),
        TokenLoc('token', 17),
        TokenLoc('weird$69', 23),
    ]
    actual = regex_tokenizer(split_tokenizer(), r'^\$\d+$', '<money>')(data)
    assert actual == expected


def test_combine_tokenizer():
    """Tests ankura.pipeline.combine_tokenizer"""
    names = io.StringIO('bob\nbill\nfred\n')
    #       0         1         2
    #       012345678901234567890123
    data = 'Lorem Ipsum Fred bob set'
    expected = [
        TokenLoc('lorem', 0),
        TokenLoc('ipsum', 6),
        TokenLoc('<name>', 12),
        TokenLoc('<name>', 17),
        TokenLoc('set', 21),
    ]
    actual = combine_tokenizer(default_tokenizer(), names, '<name>')(data)
    assert actual == expected


def test_stopword_tokenizer():
    """Tests ankura.pipeline.stopword_tokenizer"""
    stopwords = io.StringIO('the\na\nthis\nof\n')
    #       0          1         2
    #       01234 567890123456789012345
    data = 'What\'s the point Of this?'
    expected = [
        TokenLoc('whats', 0),
        TokenLoc('point', 11),
    ]
    actual = stopword_tokenizer(default_tokenizer(), stopwords)(data)
    assert actual == expected


def test_frequency_tokenizer():
    """Tests ankura.pipeline.frequency_tokenizer"""
    disk = [
        #          0         1
        #          012345678901234567
        Text("1", "rare normal common"),
        Text("2", "normal common"),
        Text("3", "normal common"),
        Text("4", "normal common"),
        Text("5", "common"),
        Text("6", "common"),
    ]
    def _mock_inputer():
        for text in disk:
            yield _mock_file(text.name, text.data)

    cases = [
        (
            2, 5,
            [
                [TokenLoc('normal', 5)],
                [TokenLoc('normal', 0)],
                [TokenLoc('normal', 0)],
                [TokenLoc('normal', 0)],
                [],
                [],
            ],
        ), (
            2, None,
            [
                [TokenLoc('normal', 5), TokenLoc('common', 12)],
                [TokenLoc('normal', 0), TokenLoc('common', 7)],
                [TokenLoc('normal', 0), TokenLoc('common', 7)],
                [TokenLoc('normal', 0), TokenLoc('common', 7)],
                [TokenLoc('common', 0)],
                [TokenLoc('common', 0)],
            ],
        ), (
            None, 5,
            [
                [TokenLoc('rare', 0), TokenLoc('normal', 5)],
                [TokenLoc('normal', 0)],
                [TokenLoc('normal', 0)],
                [TokenLoc('normal', 0)],
                [],
                [],
            ],
        ),
    ]

    for rare, common, expecteds in cases:
        pipeline = Pipeline(
            _mock_inputer,
            whole_extractor(),
            split_tokenizer(),
            noop_labeler(),
        )
        pipeline.tokenizer = frequency_tokenizer(pipeline, rare, common)
        for text, expected in zip(disk, expecteds):
            actual = pipeline.tokenizer(text.data)
            assert actual == expected, (text.data, rare, common)


def test_noop_labeler():
    """Tests ankura.pipeline.noop_labeler"""
    assert noop_labeler()('name') == {}


def test_title_labeler():
    """Tests ankura.pipeline.title_labeler"""
    assert title_labeler()('name') == {'title': 'name'}


def test_dir_labeler():
    """Tests ankura.pipeline.dir_labeler"""
    assert dir_labeler()('dirname/filename') == {'dirname': 'dirname'}


def test_composite_labeler():
    """Tests ankura.pipeline.composite_labeler"""
    labeler = composite_labeler(title_labeler(), dir_labeler())
    assert labeler('a/b') == {'title': 'a/b', 'dirname': 'a'}

"""Provides access to some standard datasets"""

import functools
import itertools
import os
import urllib.request

import ankura


download_dir = os.path.join(os.getenv('HOME'), '.ankura') # pylint: disable=invalid-name

def _path(name):
    return os.path.join(download_dir, name)


base_url = 'https://github.com/jlund3/data/raw/data2/' # pylint: disable=invalid-name

def _url(name):
    return base_url + name


def _ensure_dir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except FileExistsError:
        pass


def _ensure_download(name):
    path = _path(name)
    if not os.path.isfile(path):
        _ensure_dir(path)
        urllib.request.urlretrieve(_url(name), path)


def open_download(name, mode='r'):
    """Gets a file object for the given name, downloading the data to download
    dir from base_url if needed. By default the files are opened in read mode.
    If used as part of an inputer, the mode should likely be changed to binary
    mode. For a list of useable names with the default base_url, see
    download_inputer.
    """
    _ensure_download(name)
    return open(_path(name), mode)


def download_inputer(*names):
    """Generates file objects for the given names, downloading the data to
    download_dir from base_url if needed. Using the default base_url the
    available names are:
        * bible/bible.txt
        * bible/xref.txt
        * newsgroups/newsgroups.tar.gz
        * stopwords/english.txt
        * stopwords/jacobean.txt
    """
    @functools.wraps(download_inputer)
    def _inputer():
        for name in names:
            yield open_download(name, mode='rb')
    return _inputer


def bible():
    """Gets a Corpus containing the King James version of the Bible with over
    250,000 cross references
    """
    pipeline = ankura.pipeline.Pipeline(
        download_inputer('bible/bible.txt'),
        ankura.pipeline.line_extractor(),
        ankura.pipeline.stopword_tokenizer(
            ankura.pipeline.default_tokenizer(),
            itertools.chain(
                open_download('stopwords/english.txt'),
                open_download('stopwords/jacobean.txt'),
            )
        ),
        ankura.pipeline.composite_labeler(
            ankura.pipeline.title_labeler(),
            ankura.pipeline.multistring_labeler(
                open_download('bible/xref.txt'),
                attr='xref',
            )
        ),
        ankura.pipeline.keep_filter(),
    )
    pipeline.tokenizer = ankura.pipeline.frequency_tokenizer(pipeline, 1)
    return pipeline.run(_path('bible.pickle'))

"""Provides access to some standard datasets"""

import functools
import os
import urllib

import ankura


download_dir = os.path.join(os.getenv('HOME'), '.ankura') # pylint: disable=invalid-name

def _path(name):
    return os.path.join(download_dir, name)


base_url = 'https://github.com/jlund3/data/raw/master/' # pylint: disable=invalid-name

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


def _open_download(name):
    _ensure_download(name)
    return open(_path(name), 'rb')


def download_inputer(*names):
    """Generates file objects for the given names, downloading the data to
    download_dir from base_url if needed. The available names are:
        * bible/bible.txt
        * bible/xref.txt
    """
    @functools.wraps(download_inputer)
    def _inputer():
        for name in names:
            yield _open_download(name)
    return _inputer


def bible():
    """Gets a Corpus containing the King James version of the Bible with over
    250,000 cross references
    """
    return ankura.pipeline.Pipeline(
        download_inputer('bible/bible.txt'),
        ankura.pipeline.line_extractor(),
        ankura.pipeline.default_tokenizer(),
        ankura.pipeline.title_labeler(),
    ).run()


def newsgroups():
    """Gets a Corpus containing roughly 20,000 messages from 20 different
    usenet groups in the early 1990's
    """
    return ankura.pipeline.Pipeline(
        download_inputer('newsgroups/newsgroups.tar.gz'),
        ankura.pipeline.skip_extractor(),
        ankura.pipeline.default_tokenizer(),
        ankura.pipeline.composite_labeler(
            ankura.pipeline.title_labeler(),
            ankura.pipeline.dir_labeler('newsgroup'),
        ),
    ).run()

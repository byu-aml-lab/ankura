"""Provides access to some standard datasets"""

import functools
import itertools
import os
import urllib.request

import ankura


download_dir = os.path.join(os.getenv('HOME'), '.ankura') # pylint: disable=invalid-name

def _path(name):
    return os.path.join(download_dir, name)


base_url = 'https://github.com/jlund3/data/raw/data2' # pylint: disable=invalid-name

def _url(name):
    return os.path.join(base_url, name)


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
            ankura.pipeline.title_labeler('verse'),
            ankura.pipeline.list_labeler(
                open_download('bible/xref.txt'),
                'xref',
            ),
        ),
        ankura.pipeline.keep_filterer(),
    )
    pipeline.tokenizer = ankura.pipeline.frequency_tokenizer(pipeline, 2)
    return pipeline.run(_path('bible.pickle'))


def newsgroups():
    """Gets a Corpus containing roughly 20,000 usenet postings from 20
    different newsgroups in the early 1990's
    """
    pipeline = ankura.pipeline.Pipeline(
        download_inputer('newsgroups/newsgroups.tar.gz'),
        ankura.pipeline.targz_extractor(
            ankura.pipeline.skip_extractor(errors='replace'),
        ),
        ankura.pipeline.remove_tokenizer(
            ankura.pipeline.stopword_tokenizer(
                ankura.pipeline.default_tokenizer(),
                open_download('stopwords/english.txt'),
            ),
            r'^(.{0,2}|.{15,})$', # remove any token t for which 2<len(t)<=15
        ),
        ankura.pipeline.composite_labeler(
            ankura.pipeline.title_labeler('id'),
            ankura.pipeline.dir_labeler('newsgroup'),
        ),
        ankura.pipeline.length_filterer(),
    )
    pipeline.tokenizer = ankura.pipeline.frequency_tokenizer(pipeline, 50, 2000)
    return pipeline.run(_path('newsgroups.pickle'))


def amazon():
    """Gets a Corpus containing roughly 40,000 Amazon product reviews, with
    star ratings.
    """
    pipeline = ankura.pipeline.Pipeline(
        download_inputer('amazon/amazon.txt'),
        ankura.pipeline.line_extractor('\t'),
        ankura.pipeline.stopword_tokenizer(
            ankura.pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        ankura.pipeline.composite_labeler(
            ankura.pipeline.title_labeler('id'),
            ankura.pipeline.float_labeler(
                open_download('amazon/amazon.stars'),
                'rating',
            ),
        ),
        ankura.pipeline.length_filterer(),
    )
    pipeline.tokenizer = ankura.pipeline.frequency_tokenizer(pipeline, 50)
    return pipeline.run(_path('amazon.pickle'))

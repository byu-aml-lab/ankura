"""Provides access to some standard downloadable datasets.

The available datasets (and corresponding import functions) include:
    * bible
    * newsgroups
    * amazon
These imports depend on two module variables which can be mutated to change the
download behavior of these imports. Downloaded and pickled data will be stored
in the path given by `download_dir`, and data will be downloaded from
`base_url`. By default, `download_dir` will be '$HOME/.ankura' while base_url
will point at a GitHub repo designed for use with 
"""

import functools
import itertools
import os
import urllib.request

from . import pipeline
import posixpath

download_dir = os.path.join(os.getenv('HOME'), '.ankura')

def _path(name):
    return os.path.join(download_dir, name)


base_url = 'https://github.com/jefflund/data/raw/data2'

def _url(name):
    return posixpath.join(base_url, name)


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
    250,000 cross references.
    """
    p= pipeline.Pipeline(
        download_inputer('bible/bible.txt'),
        pipeline.line_extractor(),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            itertools.chain(
                open_download('stopwords/english.txt'),
                open_download('stopwords/jacobean.txt'),
            )
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('verse'),
            pipeline.list_labeler(
                open_download('bible/xref.txt'),
                'xref',
            ),
        ),
        pipeline.keep_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 2)
    return p.run(_path('bible.pickle'))


def newsgroups():
    """Gets a Corpus containing roughly 20,000 usenet postings from 20
    different newsgroups in the early 1990's.
    """
    coarse_mapping = {
        'comp.graphics': 'comp',
        'comp.os.ms-windows.misc': 'comp',
        'comp.sys.ibm.pc.hardware': 'comp',
        'comp.sys.mac.hardware': 'comp',
        'comp.windows.x': 'comp',
        'rec.autos': 'rec',
        'rec.motorcycles': 'rec',
        'rec.sport.baseball': 'rec',
        'rec.sport.hockey': 'rec',
        'sci.crypt': 'sci',
        'sci.electronics': 'sci',
        'sci.med': 'sci',
        'sci.space': 'sci',
        'misc.forsale': 'misc',
        'talk.politics.misc': 'politics',
        'talk.politics.guns': 'politics',
        'talk.politics.mideast': 'politics',
        'talk.religion.misc' : 'religion',
        'alt.atheism' : 'religion',
        'soc.religion.christian' : 'religion',
    }

    p = pipeline.Pipeline(
        download_inputer('newsgroups/newsgroups.tar.gz'),
        pipeline.targz_extractor(
            pipeline.skip_extractor(errors='replace'),
        ),
        pipeline.remove_tokenizer(
            pipeline.stopword_tokenizer(
                pipeline.default_tokenizer(),
                itertools.chain(open_download('stopwords/english.txt'),
                                open_download('stopwords/newsgroups.txt'))
            ),
            r'^(.{0,2}|.{15,})$', # remove any token t for which 2<len(t)<=15
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.dir_labeler('newsgroup'),
            lambda n: {'coarse_newsgroup': coarse_mapping[os.path.dirname(n)]},
        ),
        pipeline.length_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 100, 2000)
    return p.run(_path('newsgroups.pickle'))


def amazon():
    """Gets a Corpus containing roughly 40,000 Amazon product reviews, with
    star ratings.
    """
    p = pipeline.Pipeline(
        download_inputer('amazon/amazon.txt'),
        pipeline.line_extractor('\t'),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.float_labeler(
                open_download('amazon/amazon.stars'),
                'rating',
            ),
        ),
        pipeline.length_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 50)
    return p.run(_path('amazon.pickle'))

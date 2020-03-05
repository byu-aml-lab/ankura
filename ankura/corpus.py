"""Provides access to some standard downloadable datasets.

The available datasets (and corresponding import functions) include:
    * bible
    * newsgroups
    * amazon
These imports depend on two module variables which can be mutated to change the
download behavior of these imports. Downloaded and pickled data will be stored
in the path given by `download_dir`, and data will be downloaded from
`base_url`. By default, `download_dir` will be '$HOME/.ankura' while base_urlwill point at a GitHub repo designed for use with
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


base_url = 'https://github.com/byu-aml-lab/data/raw/data2'

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


def _binary_labeler(data, threshold,
                    attr='label', delim='\t',
                    needs_split=True):
    stream = data
    if needs_split:
        stream = (line.rstrip(os.linesep).split(delim, 1) for line in stream)
    stream = ((key, float(value) >= threshold) for key, value in stream)
    return pipeline.stream_labeler(stream, attr)


def _binary_string_labeler(data, threshold,
                           attr='binary_rating', delim='\t',
                           needs_split=True):
    stream = data
    if needs_split:
        stream = (line.rstrip(os.linesep).split(delim, 1) for line in stream)
    stream = ((key, 'positive' if float(value) >= threshold else 'negative')
               for key, value in stream)
    return pipeline.stream_labeler(stream, attr)


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

def tripadvisor():
    """Gets a corpus containing hotel reviews on trip advisor with ~240,000 documents"""
    # These words should be removed, as they don't end up cooccuring with other
    # words
    extra_stopwords = ['zentral', 'jederzeit', 'gerne', 'gutes', 'preis',
    'plac茅', 'empfehlenswert', 'preisleistungsverh盲ltnis', 'posizione',
    'sch枚nes', 'zu', 'empfehlen', 'qualit茅prix', 'tolles', 'rapporto',
    'guter', 'struttura']

    label_stream = []

    def regex_extractor(docfile):
        import re

        text = docfile.read().decode('utf-8')

        documents = re.findall('<Content>(.*$)', text, re.M)
        labels = re.findall('<Overall>(\d*)', text, re.M)

        for i in range(len(documents)):
            overall = int(labels[i])
            label = overall if overall == 5 else 0
            label_stream.append((str(i), label))
            yield pipeline.Text(str(i), documents[i])

    p = pipeline.Pipeline(
        download_inputer('tripadvisor/tripadvisor.tar.gz'),
        pipeline.targz_extractor(regex_extractor),
        pipeline.stopword_tokenizer(
                pipeline.default_tokenizer(),
                itertools.chain(
                    open_download('stopwords/english.txt'),
                    extra_stopwords
                )
        ),
        pipeline.composite_labeler(
            pipeline.stream_labeler(label_stream),
            _binary_string_labeler(
                label_stream,
                5,
                'binary_rating',
                needs_split=False,
            ),
        ),
        pipeline.length_filterer(30),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 150)
    return p.run(_path('tripadvisor.pickle'))


def yelp():
    """ Gets a corpus containing Yelp reviews with 25431 documents """

    base_tokenzer = pipeline.default_tokenizer()

    def strip_non_alpha(token):
        return ''.join(c for c in token if c.isalnum())

    def tokenizer(data):
        tokens = base_tokenzer(data)
        tokens = [pipeline.TokenLoc(strip_non_alpha(t.token), t.loc) for t in tokens]
        tokens = [t for t in tokens if t.token]
        return tokens

    p = pipeline.Pipeline(
        download_inputer('yelp/yelp.txt'),
        pipeline.line_extractor('\t'),
        pipeline.stopword_tokenizer(
            tokenizer,
            open_download('stopwords/english.txt'),
        ),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.float_labeler(
                open_download('yelp/yelp.response'),
                'rating',
            ),
            _binary_string_labeler(
                open_download('yelp/yelp.response'),
                5,
                'binary_rating',
            ),
        ),
        pipeline.length_filterer(30),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 50)
    return p.run(_path('yelp.pickle'))


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

def toy():
    p = pipeline.Pipeline(
        download_inputer('toy/toy.tar.gz'),
        pipeline.targz_extractor(
            pipeline.whole_extractor()
        ),
        pipeline.default_tokenizer(),
        pipeline.composite_labeler(
            pipeline.title_labeler('id'),
            pipeline.dir_labeler('directory')
        ),
        pipeline.length_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p)
    return p.run(_path('toy.pickle'))

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
            r'^(.{0,2}|.{15,})$', # remove any token t with len(t)<=2 or len(t)>=15
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

def amazon_medium():
    """Gets a corpus containing 100,000 Amazon product reviews, with star ratings.
    """
    label_stream = BufferedStream()

    def label_extractor(docfile, value_key='reviewText', label_key='overall'):

        import json

        for i, line in enumerate(docfile):
            line = json.loads(line.decode('utf-8'))
            label_stream.append((str(i), line[label_key]))

            yield pipeline.Text(str(i), line[value_key])

    p = pipeline.Pipeline(
        download_inputer('amazon_medium/amazon_medium.json.gz'),
        pipeline.gzip_extractor(label_extractor),
        pipeline.stopword_tokenizer(
            pipeline.default_tokenizer(),
            open_download('stopwords/english.txt'),
        ),
        pipeline.stream_labeler(label_stream),
        pipeline.length_filterer(),
    )

    p.tokenizer = pipeline.frequency_tokenizer(p, 100, 2000)
    return p.run(_path('amazon_medium.pickle'))

def amazon():
    """Gets a Corpus containing roughly 40,000 Amazon product reviews, with
    star ratings.
    """
    def binary_labeler(data, threshold, attr='label', delim='\t'):
        stream = (line.rstrip(os.linesep).split(delim, 1) for line in data)
        stream = ((key, float(value) >= threshold) for key, value in stream)
        return pipeline.stream_labeler(stream, attr)

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
            _binary_string_labeler(
                open_download('amazon/amazon.stars'),
                5,
                'binary_rating',
            ),
        ),
        pipeline.length_filterer(),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 50)
    return p.run(_path('amazon.pickle'))

def congress():
    """Corpus on congress talking about different issues/bills."""
    def congress_labeler(title):
        # See .ankura/congress.README.v1.1.txt for more info
        # ###_@@@@@@_%%%%$$$_PMV
        # P - party (D, R, or X)
        # V - vote indicator
        return {'title': title,
                'party': title[19],
                'vote' : title[21]}

    def party_filterer():
        """Remove docs with Independent speakers (26 docs before other filters)"""
        accepted = {'D', 'R'}
        def _filterer(doc):
            return doc.metadata['party'] in accepted
        return _filterer

    p = pipeline.Pipeline(
        download_inputer('congress/congress.tar.gz'),
        pipeline.targz_extractor(
            pipeline.whole_extractor()
        ),
        pipeline.stopword_tokenizer(
                pipeline.default_tokenizer(),
                open_download('stopwords/english.txt'),
        ),
        congress_labeler,
        pipeline.composite_filterer(
            pipeline.length_filterer(50),
            party_filterer()
        )
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 50)
    return p.run(_path('congress.pickle'))


# TIME 3760.985
def nsfabstracts():
    import re

    def extensional_extractor(base_extractor, extensions=['.txt']):
        def _extractor(docfile):
            if os.path.splitext(docfile.name)[1] not in extensions:
                return
            else:
                yield from base_extractor(docfile)
        return _extractor

    def metadata_stream_extractor(metadata_stream):
        prgrm_num = re.compile(r'\d+')
        prgrm_name = re.compile(r'\d+\s+(.*)')

        def _extractor(docfile):
            text = docfile.read().decode('utf-8')
            metadata = {}
            abstract_text = []

            collecting = False
            for line in text.split('\n'):
                if collecting:
                    abstract_text.append(line)

                elif line.startswith('Abstract    :'):
                    collecting = True

                elif line.startswith('NSF Program :'):
                    program_line = line[len('NSF Program :'):]

            metadata['program_number'] = int(prgrm_num.findall(program_line)[0])
            metadata['program_name'] = prgrm_name.findall(program_line)[0]
            metadata['filename'] = docfile.name

            abstract_text = (' '.join(t.strip() for t in abstract_text)).strip()

            metadata_stream.append((docfile.name, metadata))
            yield pipeline.Text(docfile.name, abstract_text)
        return _extractor

    def metadata_stream_labeler(stream):
        cache = {}
        def _labeler(name):
            if name in cache:
                return cache.pop(name)
            for key, value in stream:
                if key == name:
                    return value
                else:
                    cache[key] = value
            raise KeyError(name)
        return _labeler

    metadata_stream = []

    p = pipeline.Pipeline(
        download_inputer('nsfabstracts/nsfabstracts.tar.gz'),
        pipeline.targz_extractor(
            extensional_extractor(
                metadata_stream_extractor(metadata_stream)
            )
        ),
        pipeline.stopword_tokenizer(
                pipeline.default_tokenizer(),
                open_download('stopwords/english.txt'),
        ),
        metadata_stream_labeler(metadata_stream),
        pipeline.length_filterer(30),
    )
    p.tokenizer = pipeline.frequency_tokenizer(p, 150)
    return p.run(_path('nsfabstracts.pickle'))


class BufferedStream(object):

    def __init__(self):
        self.buf = []

    def append(self, key_value):
        self.buf.append(key_value)

    def __iter__(self):
        while self.buf:
            tup = self.buf.pop()
            key = tup[0]
            val = tup[1]
            yield key, val

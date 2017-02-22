"""Functionality for importing datasets from disk"""

import functools
import collections
import glob
import gzip
import os
import string

# Types used throughout the pipeline process

Text = collections.namedtuple('Text', ['name', 'data'])
TokenLoc = collections.namedtuple('TokenLoc', ['token', 'loc'])
TypeLoc = collections.namedtuple('TypeLoc', ['type', 'loc'])
Document = collections.namedtuple('Document', ['text', 'types', 'metadata'])
Corpus = collections.namedtuple('Corpus', ['documents', 'vocabulary'])


# Inputers are callables which generate the filenames a pipeline should read


def file_inputer(*filenames):
    """Gets file objects for each of the given filenames"""
    @functools.wraps(file_inputer)
    def _inputer():
        for filename in filenames:
            yield open(filename, 'rb')
    return _inputer


def glob_inputer(pattern):
    """Gets file objects for each filename found using a glob pattern"""
    @functools.wraps(glob_inputer)
    def _inputer():
        return file_inputer(*glob.glob(pattern))
    return _inputer


# Extractors are callables which generate Text from a file object.


def whole_extractor():
    """Extracts the entire contents of a file as a Text"""
    @functools.wraps(whole_extractor)
    def _extractor(docfile):
        yield Text(docfile.name, docfile.read().decode())
    return _extractor


def skip_extractor(delim='\n\n'):
    """Extracts the contents of a file as a Text after skipping a header"""
    @functools.wraps(skip_extractor)
    def _extractor(docfile):
        data = docfile.read().decode()
        start = data.index(delim) + len(delim)
        yield Text(docfile.name, data[start:])
    return _extractor


def line_extractor(delim=' '):
    """Treats each line of a file as a Text"""
    @functools.wraps(line_extractor)
    def _extractor(docfile):
        for line in docfile:
            line = line.decode()
            index = line.index(delim)
            yield Text(line[:index], line[index + len(delim):])
    return _extractor


def gzip_extractor(base_extractor):
    """Passes the uncompressed contents of a docfile to a base extractor"""
    @functools.wraps(gzip_extractor)
    def _extractor(docfile):
        return base_extractor(gzip.GzipFile(fileobj=docfile))
    return _extractor


# Tokenizers are callables which split Text data into TokenLoc.


def split_tokenizer(delims=string.whitespace):
    """Splits a text on delimiting characters"""
    @functools.wraps(split_tokenizer)
    def _tokenizer(data):
        tokens = []
        begin = -1 # Set to -1 when looking for start of token.
        for i, char in enumerate(data):
            if char in delims:
                if begin >= 0:
                    tokens.append(TokenLoc(data[begin: i], begin))
                    begin = -1
            elif begin == -1:
                begin = i
        if begin >= 0: # Last token might be at EOF.
            tokens.append(TokenLoc(data[begin:], begin))
        return tokens
    return _tokenizer


def default_transform(token):
    """Default transform for transform_tokenizer"""
    return ''.join(c.lower() for c in token if c.isalpha())


def transform_tokenizer(transform=default_transform, delims=string.whitespace):
    """Splits a text on delimiting characters and transforms the tokens"""
    splitter = split_tokenizer(delims)
    @functools.wraps(transform_tokenizer)
    def _tokenizer(data):
        tokens = splitter(data)
        tokens = [TokenLoc(transform(t.token), t.loc) for t in tokens]
        tokens = [t for t in tokens if t.token]
        return tokens
    return _tokenizer


# Labelers are callables which generate metadata from a Text name.

def noop_labeler():
    """Returns an empty labeling"""
    @functools.wraps(noop_labeler)
    def _labeler(_name):
        return {}
    return _labeler


def title_labeler(attr='title'):
    """Returns a labeling with the name as the value"""
    @functools.wraps(title_labeler)
    def _labeler(name):
        return {attr: name}
    return _labeler


def dir_labeler(attr='dirname'):
    """Returns a labeling with the dirname of the name as the value"""
    @functools.wraps(dir_labeler)
    def _labeler(name):
        return {attr: os.path.dirname(name)}
    return _labeler


def composite_labeler(*labelers):
    """Returns a labeling with the merged results of several labelers"""
    @functools.wraps(composite_labeler)
    def _labeler(name):
        labels = {}
        for labeler in labelers:
            labels.update(labeler(name))
        return labels
    return _labeler


# Pipeline puts the previous pieces together to import a Corpus


class VocabBuilder(object):
    """Stores a bidirectional map of token to token ids"""

    def __init__(self):
        self.tokens = []
        self.types = {}

    def __getitem__(self, token):
        if token not in self.types:
            self.types[token] = len(self.tokens)
            self.tokens.append(token)
        return self.types[token]

    def convert(self, tokens):
        """Converts a sequence of TokenLoc to a sequence of TypeLoc"""
        return [TypeLoc(self[t.token], t.loc) for t in tokens]


class Pipeline(object):
    """Pipeline"""

    def __init__(self, inputer, extractor, tokenizer, labeler):
        self.inputer = inputer
        self.extractor = extractor
        self.tokenizer = tokenizer
        self.labeler = labeler

    def run(self):
        """Creates a new Corpus using the Pipeline"""
        documents = []
        vocab = VocabBuilder()
        for docfile in self.inputer():
            for text in self.extractor(docfile):
                tokens = self.tokenizer(text.data)
                types = vocab.convert(tokens)
                metadata = self.labeler(text.name)
                documents.append(Document(text.data, types, metadata))
        return Corpus(documents, vocab.tokens)

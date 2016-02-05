"""Functions for creating data import pipelines

An import usually consists of a read followed by a chain of transformations.
For example, a typical import could look like:
    dataset = read_glob('newsgroups/*/*', tokenizer=tokenize.news)
    dataset = filter_stopwords(dataset, 'stopwords/english.txt')
    dataset = filter_rarewords(dataset, 20)

Alternatively, a pipeline can be created programatically and then run in the
following way:
    pipeline = [(read_glob, 'newsgroups/*/*', tokenize.news),
                (filter_stopwords, 'stopwords/english.txt'),
                (filter_rarewords, 20)]
    dataset = run_pipeline(pipeline)
"""
import io
import glob
import numpy
import random
import scipy.sparse

from . import tokenize
# from ankura import tokenize


class Dataset(object):
    """Stores a bag-of-words dataset

    The dataset should be considered immutable. Consequently, dataset
    attributes are accessible only through properties which have no setters.
    The docwords matrix will be a sparse scipy matrix of uint. The vocab and
    titles will both be lists of str. The cooccurrences matrix will be a numpy
    array of float.
    """
    def __init__(self, docwords, vocab, titles):
        self._docwords = docwords
        self._vocab = vocab
        self._titles = titles
        self._cooccurrences = None
        self._tokens = {}
        # TODO add ability to add document metadata or labels

    @property
    def M(self):
        """Gets the sparse docwords matrix"""
        return self._docwords

    @property
    def docwords(self):
        """Gets the sparse docwords matrix"""
        return self.M

    @property
    def vocab(self):
        """Gets the list of vocabulary items"""
        return self._vocab

    @property
    def titles(self):
        """Gets the titles of each document"""
        return self._titles

    @property
    def Q(self):
        """Gets the word cooccurrence matrix"""
        # TODO add ways to augment Q with additional labeled data
        if self._cooccurrences is None:
            self._compute_cooccurrences()
        return self._cooccurrences

    @property
    def cooccurrences(self):
        """Gets the word cooccurrence matrix"""
        return self.Q

    def _compute_cooccurrences(self):
        # See supplementary 4.1 of Aurora et. al. 2012 for information on these
        vocab_size, num_docs = self.M.shape
        H_tilde = scipy.sparse.csc_matrix(self.M, dtype=float)
        H_hat = numpy.zeros(vocab_size)

        # Construct H_tilde and H_hat
        for j in range(H_tilde.indptr.size - 1):
            # get indices of column j
            col_start = H_tilde.indptr[j]
            col_end = H_tilde.indptr[j + 1]
            row_indices = H_tilde.indices[col_start: col_end]

            # get count of tokens in column (document) and compute norm
            count = numpy.sum(H_tilde.data[col_start: col_end])
            norm = count * (count - 1)

            # update H_hat and H_tilde (see supplementary)
            if norm != 0:
                H_hat[row_indices] = H_tilde.data[col_start: col_end] / norm
                H_tilde.data[col_start: col_end] /= numpy.sqrt(norm)

        # construct and store normalized Q
        Q = H_tilde * H_tilde.transpose() - numpy.diag(H_hat)
        self._cooccurrences = numpy.array(Q / num_docs)

    @property
    def vocab_size(self):
        """Gets the size of the dataset vocabulary"""
        return self._docwords.shape[0]

    @property
    def num_docs(self):
        """Gets the number of documents in the dataset"""
        return self._docwords.shape[1]

    def doc_tokens(self, doc_id, rng=random):
        """Converts a document from counts to a sequence of token ids

        The conversion for any one document is only computed once, and the
        resultant tokens are shuffled. However, the computations are performed
        lazily.
        """
        if doc_id in self._tokens:
            return self._tokens[doc_id]

        token_ids, _, counts = scipy.sparse.find(self._docwords[:, doc_id])
        tokens = []
        for token_id, count in zip(token_ids, counts):
            tokens.extend([token_id] * count)
        rng.shuffle(tokens)

        self._tokens[doc_id] = tokens
        return tokens


def read_uci(docwords_filename, vocab_filename):
    """Reads a Dataset from disk in UCI bag-of-words format

    The docwords file is expected to have the following format:
    ---
    D
    W
    NNZ
    docId wordId count
    docId wordId count
    docId wordId count
    ...
    docId wordId count
    docId wordId count
    docId wordId count
    ---
    where D is the number of documents, W is the number of word types in the
    vocabulary, and NNZ is the number of non-zero counts in the data. Each
    subsequent row is a triple consisting of a document id, a word id, and a
    non-zero count indicating the number of occurences of the word in the
    document. Note that both the document id and the word id are
    one-indexed.

    The vocab file is expected to have the actual tokens of the vocabulary.
    There is one token per line, with the line numbers corresponding to the
    word ids in the docwords file.

    Since the uci format does not give any additional information about
    documents, we make the titles simply the string version of the docId's.
    """
    # read in the vocab file
    vocab = []
    with open(vocab_filename) as vocab_file:
        for line in vocab_file:
            vocab.append(line.strip())

    # read in the docwords file
    with open(docwords_filename) as docwords_file:
        num_docs = int(docwords_file.readline())
        num_words = int(docwords_file.readline())
        docwords_file.readline() # ignore nnz

        docwords = scipy.sparse.lil_matrix((num_words, num_docs), dtype='uint')
        for line in docwords_file:
            doc, word, count = (int(x) for x in line.split())
            docwords[word - 1, doc - 1] = count

    # construct and return the Dataset
    titles = [str(i) for i in range(num_docs)]
    return Dataset(docwords.tocsc(), vocab, titles)


def _build_dataset(docdata, tokenizer):
    # read each file, tracking vocab and word counts
    docs = []
    vocab = {}
    titles = []
    for title, data in docdata:
        doc = {}
        for token in tokenizer(data):
            if token not in vocab:
                vocab[token] = len(vocab)
            token_id = vocab[token]
            doc[token_id] = doc.get(token_id, 0) + 1
        docs.append(doc)
        titles.append(title)

    # construct the docword matrix using the vocab map
    docwords = scipy.sparse.lil_matrix((len(vocab), len(docs)), dtype='uint')
    for doc, counts in enumerate(docs):
        for word, count in counts.items():
            docwords[word, doc] = count

    # convert vocab from a token to index map into a list of tokens
    vocab = {index: token for token, index in vocab.items()}
    vocab = [vocab[index] for index in range(len(vocab))]

    # construct and return the Dataset
    return Dataset(docwords.tocsc(), vocab, titles)


def read_glob(glob_pattern, tokenizer=tokenize.simple):
    """Read a Dataset from a set of files found by a glob pattern

    Each file found by the glob pattern corresponds to a single document in the
    dataset. Each file object is then tokenized by the provided tokenizer
    function. The document titles are given by the corresponding filenames.
    """
    filenames = glob.glob(glob_pattern)
    docdata = ((name, open(name, errors='replace')) for name in filenames)
    return _build_dataset(docdata, tokenizer)


def read_file(filename, tokenizer=tokenize.simple):
    """Read a Dataset from one file containing one document per line

    For each line of the file, the first sequence of letters unbroken by
    whitespace will be considered the title of the document.  All of the words
    after the first non-newline whitespace sequence will be considered words of
    the document
    """
    lines = (line.split(None, 1) for line in open(filename))
    docdata = ((title, io.StringIO(doc)) for title, doc in lines)
    return _build_dataset(docdata, tokenizer)


def _filter_vocab(dataset, filter_func):
    """Filters out a set of stopwords based on a filter function"""
    # track which vocab indices should be discarded and which should be kept
    stop_index = []
    keep_index = []
    for i, word in enumerate(dataset.vocab):
        if filter_func(i, word):
            keep_index.append(i)
        else:
            stop_index.append(i)

    # construct dataset with filtered docwords and vocab
    docwords = dataset.docwords[keep_index, :]
    vocab = scipy.delete(dataset.vocab, stop_index)
    return Dataset(docwords, vocab.tolist(), dataset.titles)


def _get_wordlist(filename, tokenizer):
    if tokenizer:
        return set(tokenizer(open(filename)))
    else:
        return {word.strip() for word in open(filename)}


def filter_stopwords(dataset, stopword_filename, tokenizer=None):
    """Filters out a set of stopwords from a dataset

    The stopwords file is expected to contain a single stopword token per line.
    The original data is unchanged.
    """
    stopwords = _get_wordlist(stopword_filename, tokenizer)
    keep = lambda i, v: v not in stopwords
    return _filter_vocab(dataset, keep)


def combine_words(dataset, combine_filename, replace, tokenizer=None):
    """Combines a set of words into a single token

    The combine file is expected to contain a single token per line. The
    original data is unchanged.
    """
    words = _get_wordlist(combine_filename, tokenizer)
    index = sorted([dataset.vocab.index(v) for v in words])
    sums = dataset.docwords[index, :].sum(axis=0)

    keep = lambda i, v: i not in index[1:]
    combined = _filter_vocab(dataset, keep)
    combined.docwords[index[0], :] = sums
    combined.vocab[index[0]] = replace
    return combined


def filter_rarewords(dataset, doc_threshold):
    """Filters rare words which do not appear in enough documents"""
    keep = lambda i, v: dataset.docwords[i, :].nnz >= doc_threshold
    return _filter_vocab(dataset, keep)


def filter_commonwords(dataset, doc_threshold):
    """Filters rare words which appear in too many documents"""
    keep = lambda i, v: dataset.docwords[i, :].nnz <= doc_threshold
    return _filter_vocab(dataset, keep)


def filter_smalldocs(dataset, token_threshold, prune_vocab=True):
    """Filters documents whose token count is less than the threshold

    After removing all short documents, the vocabulary can optionally be pruned
    so that if all documents containing a particular token, that token will
    also be removed from the vocabulary. By default, prune_vocab is True.
    """
    token_counts = dataset.docwords.sum(axis=0)
    keep_index = []
    stop_index = []
    for i, count in enumerate(token_counts.flat):
        if count < token_threshold:
            stop_index.append(i)
        else:
            keep_index.append(i)

    docwords = dataset.docwords[:, keep_index]
    titles = scipy.delete(dataset.titles, stop_index)
    dataset = Dataset(docwords, dataset.vocab, titles)

    if prune_vocab:
        return filter_rarewords(dataset, 1)
    else:
        return dataset


def convert_docwords(dataset, conversion):
    """Applies a transformation to the docwords matrix of a dataset

    The most typical usage of this function will be to change the format of the
    docwords matrix. For example, one could change the format from the default
    lil matrix to a csc matrix with:
    dataset = convert_docwords(dataset, scipy.sparse.csc_matrix)
    """
    return Dataset(conversion(dataset.docwords), dataset.vocab, dataset.titles)


def pregenerate_doc_tokens(dataset):
    """Pregenerates the doc tokens for each document in the dataset

    In addition to generating the doc tokens for the entire dataset, this
    function returns the original dataset so that it can be used inside a
    pipeline.
    """
    for doc in range(dataset.num_docs):
        dataset.doc_tokens(doc)
    return dataset


def _prepare_split(dataset, indices):
    split_docwords = dataset.docwords[:, indices]
    split_titles = [dataset.titles[i] for i in indices]
    return Dataset(split_docwords, dataset.vocab, split_titles)


def train_test_split(dataset, train_percent=.75, rng=random):
    """Splits a dataset into training and test sets

    The train_percent gives the percent of the documents which should be used
    for training. The remaining are placed in test. Both sets will share the
    same vocab after the split, but the vocabulary is pruned so that words
    which only appear in test are discarded.
    """
    # TODO make split preserve any lazily computed things like doc tokens

    # find the indices of the docs for both train and test
    shuffled_docs = range(dataset.num_docs)
    rng.shuffle(shuffled_docs)
    split = int(len(shuffled_docs) * train_percent)
    train_docs, test_docs = shuffled_docs[:split], shuffled_docs[split:]

    # split the datasets into train and test
    train_data = _prepare_split(dataset, train_docs)
    test_data = _prepare_split(dataset, test_docs)

    # filter out words which only appear in test
    keep = lambda i, v: train_data.docwords[i, :].nnz > 0
    return _filter_vocab(train_data, keep), _filter_vocab(test_data, keep)


def run_pipeline(pipeline, append_pregenerate=True):
    """Runs an import pipeline consisting of a sequence of instructions

    Each instruction in the sequence should consist of another sequence giving
    a callable along with any applicable arguments. Each instruction should
    return a Dataset giving the docwords matrix and the vocab. Each instruction
    after the first takes the Dataset from the previous stage as the first
    argument.

    For example, one could construct an import pipeline in the following way:
    pipeline = [(read_glob, 'newsgroups/*/*', tokenize.news),
                (filter_stopwords, 'stopwords/english.txt'),
                (filter_rarewords, 20)]
    dataset = run_pipeline(pipeline)
    """
    read, transformations = pipeline[0], pipeline[1:]
    dataset = read[0](*read[1:])
    for transform in transformations:
        try:
            dataset = transform[0](dataset, *transform[1:])
        except TypeError:
            if not callable(transform):
                raise
            dataset = transform(dataset)

    if append_pregenerate:
        dataset = pregenerate_doc_tokens(dataset)
    return dataset


def get_word_cooccurrences(dataset):
    """Generates a new matrix by taking the minimum value of each row
    representing word frequency in the original docwords matrix, thereby
    creating a word cooccurrence matrix.
    For example:
                    doc1 doc2
                cat  1    1
                dog  2    2
    becomes:
                    doc1 doc2
            cat-dog  1    1
    """
    #Calculates the size of the new matrix
    size = int((dataset.vocab_size * (dataset.vocab_size - 1)) / 2)
    row = 0
    docsize = len(dataset.titles)
    old_matrix = dataset.M
    vocab_size = len(dataset.vocab)
    new_dataset = scipy.sparse.lil_matrix((size, docsize))

    #Compares each row in original matrix to the ones that come after
    for wordi in range(vocab_size):
        for wordj in range(wordi + 1,vocab_size):
            for doc in range(docsize):
                new_dataset[row, doc] = min(old_matrix[wordi, doc],
                                            old_matrix[wordj, doc])

            #row variable determines what row we're inserting
            #into in the new matrix
            row+=1
    vocab = dataset.vocab
    docs = []

    #Determines if new matrix is empty to set documents accordingly
    if size > 0:
        docs = dataset.titles

    new_vocab = []

    #Generates new vocab list for new matrix
    for i in range(0, len(vocab)):
        for j in range(i + 1, len(vocab)):
            newVocab.append(vocab[i] + '-' + vocab[j])

    #Creates new dataset based off of the calculated matrix and vocab
    result = Dataset(new_dataset, newVocab, docs)
    result = filter_rarewords(result, 1)
    return result;

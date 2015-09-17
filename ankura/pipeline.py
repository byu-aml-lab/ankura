"""Functions for creating data import pipelines

An import usually consists of a read followed by a chain of transformations.
For example, a typically import could look like:
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
import glob
import numpy
import scipy.sparse

from ankura import tokenize


class Dataset(object):
    """Stores a bag-of-words dataset"""

    def __init__(self, docwords, vocab):
        self._docwords = docwords
        self._vocab = vocab
        self._cooccurrences = None

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
        H_tilde = self.M.tocsc() # csc conversion so we can use indptr
        H_hat = numpy.zeros(vocab_size)

        # Construct H_tilde and H_hat
        for j in xrange(H_tilde.indptr.size - 1):
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

        docwords = scipy.sparse.lil_matrix((num_words, num_docs))
        for line in docwords_file:
            doc, word, count = (int(x) for x in line.split())
            docwords[word - 1, doc - 1] = count

    # construct and return the Dataset
    return Dataset(docwords, vocab)


def read_glob(glob_pattern, tokenizer=tokenize.simple):
    """Read a Dataset from a set of files found by a glob pattern"""
    # read each file, tracking vocab and word counts
    vocab = {}
    docs = []
    for filename in glob.glob(glob_pattern):
        with open(filename) as doc_file:
            tokens = tokenizer(doc_file)
            doc = {}
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
                token_id = vocab[token]
                doc[token_id] = doc.get(token_id, 0) + 1
            docs.append(doc)

    # construct the docword matrix using the vocab map
    docwords = scipy.sparse.lil_matrix((len(vocab), len(docs)))
    for doc, counts in enumerate(docs):
        for word, count in counts.iteritems():
            docwords[word, doc] = count

    # convert vocab from a token to index map into a list of tokens
    vocab = {index: token for token, index in vocab.items()}
    vocab = [vocab[index] for index in xrange(len(vocab))]

    # construct and return the Dataset
    return Dataset(docwords, vocab)



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
    return Dataset(docwords, vocab)


def filter_stopwords(dataset, stopword_filename):
    """Filters out a set of stopwords from a dataset

    The stopwords file is expected to contain a single stopword token per line.
    The original data is unchanged.
    """
    stopwords = {word.strip() for word in open(stopword_filename)}
    keep = lambda i, v: v not in stopwords
    return _filter_vocab(dataset, keep)


def filter_rarewords(dataset, doc_threshold):
    """Filters rare words which do not appear in enough documents"""
    keep = lambda i, v: dataset.docwords[i, :].nnz >= doc_threshold
    return _filter_vocab(dataset, keep)


def filter_commonwords(dataset, doc_threshold):
    """Filters rare words which appear in too many documents"""
    keep = lambda i, v: dataset.docwords[i, :].nnz <= doc_threshold
    return _filter_vocab(dataset, keep)


def run_pipeline(pipeline):
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
        dataset = transform[0](dataset, *transform[1:])
    return dataset

"""Functions for creating data import pipelines

An import usually consists of a read followed by a chain of transformations.
For example, a typically import could look like:
    corpus, vocab = read_glob('newsgroups/*/*', tokenizer=tokenize.news)
    corpus, vocab = filter_stopwords(docwords, vocab, 'stopwords/english.txt')
    corpus, vocab = filter_rarewords(docwords, vocab, 20)

Alternatively, a pipeline can be created programatically and then run in the
following way:
    pipeline = [(read_glob, 'newsgroups/*/*', tokenize.news),
                (filter_stopwords, 'stopwords/english.txt'),
                (filter_rarewords, 20)]
    docwords, vocab = run_pipeline(pipeline)
"""
import glob
import scipy.sparse

from ankura import tokenize


def read_uci(docwords_filename, vocab_filename):
    """Reads a dataset from disk in UCI bag-of-words format

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
    document. Note that both the document id and the word id are one-indexed.

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

    # docwords matrix and vocab list
    return docwords, vocab


def read_glob(glob_pattern, tokenizer=tokenize.simple):
    """Read each file from a glob as a document"""
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

    # docwords matrix and vocab list
    return docwords, vocab


# TODO add ability to read in document label data


def _filter_vocab(docwords, vocab, filter_func):
    """Filters out a set of stopwords based on a filter function"""
    # track which vocab indices should be discarded and which should be kept
    stop_index = []
    keep_index = []
    for i, word in enumerate(vocab):
        if filter_func(i, word):
            keep_index.append(i)
        else:
            stop_index.append(i)

    # return filtered docwords and vocab
    return docwords[keep_index, :], scipy.delete(vocab, stop_index)


def filter_stopwords(docwords, vocab, stopword_filename):
    """Filters out a set of stopwords from a dataset

    The stopwords file is expected to contain a single stopword token per line.
    The original data is unchanged.
    """
    stopwords = {word.strip() for word in open(stopword_filename)}
    keep = lambda i, v: v not in stopwords
    return _filter_vocab(docwords, vocab, keep)


def filter_rarewords(docwords, vocab, doc_threshold):
    """Filters rare words which do not appear in enough documents"""
    keep = lambda i, v: docwords[i, :].nnz >= doc_threshold
    return _filter_vocab(docwords, vocab, keep)


def filter_commonwords(docwords, vocab, doc_threshold):
    """Filters rare words which appear in too many documents"""
    keep = lambda i, v: docwords[i, :].nnz <= doc_threshold
    return _filter_vocab(docwords, vocab, keep)


def run_pipeline(pipeline):
    """Runs an import pipeline consisting of a sequence of instructions

    Each instruction in the sequence should consist of another sequence giving
    a callable along with any applicable arguments. Each instruction should
    return a tuple giving the docwords matrix and the vocab. Each instruction
    after the first takes the docwords matrix and vocab as the first two
    arguments by default.

    For example, one could construct an import pipeline in the following way:
    pipeline = [(read_glob, 'newsgroups/*/*', tokenize.news),
                (filter_stopwords, 'stopwords/english.txt'),
                (filter_rarewords, 20)]
    docwords, vocab = run_pipeline(pipeline)
    """
    read, transformations = pipeline[0], pipeline[1:]
    docwords, vocab = read[0](*read[1:])
    for transform in transformations:
        docwords, vocab = transform[0](docwords, vocab, *transform[1:])
    return docwords, vocab

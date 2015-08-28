"""Functions for creating data import pipelines

An import usually consists of a read followed by a chain of transformations.
For example, a typically import could look like:
    corpus, vocab = read_glob('newsgroups/*/*', tokenizer=tokenize_news)
    corpus, vocab = filter_stopwords(docwords, vocab, stopwords/english.txt')
    corpus, vocab = filter_rarewords(docwords, vocab, 20)
"""
import glob
import io
import re
import scipy.sparse


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


def tokenize_simple(doc_file):
    """A basic tokenizer which splits and filters text without preprocessing"""
    tokens = doc_file.read().split()
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    tokens = [token for token in tokens if 2 < len(tokens) < 10]
    return tokens


def tokenize_news(doc_file, tokenizer=tokenize_simple):
    """Tokenizes after skipping a file header

    Using the format from the well-known 20 newsgroups dataset, we consider the
    header to be everything before the first empty line in the file. The
    remaining contents of the file are then tokenized.
    """
    # skip header by finding first empty line
    line = doc_file.readline()
    while line.strip():
        line = doc_file.readline()
    # use tokenizer on what remains in file
    return tokenizer(doc_file)


def tokenize_html(doc_file, tokenizer=tokenize_simple):
    """Tokenizes by extracting text from an HTML file"""
    # parse the text in the html
    text = doc_file.read().strip()
    text = re.sub(r'(?is)<(script|style).*?>.*?(</\1>)', '', text)
    text = re.sub(r'(?s)<!--(.*?)-->[\n]?', '', text)
    text = re.sub(r'(?s)<.*?>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'  ', ' ', text)
    text = re.sub(r'  ', ' ', text)
    text = text.strip()
    # tokenize the parsed text
    return tokenizer(io.StringIO(text))


def read_glob(glob_pattern, tokenizer=tokenize_simple):
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
                doc[token] = doc.get(token, 0) + 1
            docs.append(doc)

    # construct the docword matrix using the vocab map
    docwords = scipy.sparse.lil_matrix((len(vocab), len(docs)))
    for doc, counts in enumerate(docs):
        for word, count in counts:
            docwords[word, doc] = count

    # convert vocab from a token to index map into a list of tokens
    vocab = {index: token for token, index in vocab.items()}
    vocab = [vocab[index] for index in range(vocab)]

    # docwords matrix and vocab list
    return docwords, vocab



def filter_stopwords(docwords, vocab, stopword_filename):
    """Filters out a set of stopwords from a dataset

    The stopwords file is expected to contain a single stopword token per line.
    The original data is unchanged.
    """
    # read in the stopwords file into a set of stopwords
    stopwords = set()
    with open(stopword_filename) as stopword_file:
        for line in stopword_file:
            stopwords.add(line.strip())

    # track which vocab indices should be discarded and which should be kept
    stop_index = []
    keep_index = []
    for i, word in enumerate(vocab):
        if word in stopwords:
            stop_index.append(i)
        else:
            keep_index.append(i)

    # remove stopwords from docwords and vocab
    docwords = docwords[keep_index, :]
    vocab = scipy.delete(vocab, stop_index)

    # docwords matrix and vocab list
    return docwords, vocab


def filter_rarewords(docwords, vocab, doc_threshold):
    """Filters rare words which do not appear in enough documents"""
    # track which vocab indices are rare and which are not
    rare_index = []
    keep_index = []
    for i in range(len(vocab)):
        if docwords[i, :].nnz < doc_threshold:
            rare_index.append(i)
        else:
            keep_index.append(i)

    # remove rare words from docwords and vocab
    docwords = docwords[keep_index, :]
    vocab = scipy.delete(vocab, rare_index)

    # docwords matrix and vocab list
    return docwords, vocab

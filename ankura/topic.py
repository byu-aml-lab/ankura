"""Functions for using and displaying topics"""

import functools
import sys

import numpy
import scipy.spatial
import sklearn.decomposition

import ankura.pipeline
import ankura.util


def topic_summary(topics, corpus=None, n=10):
    """Gets the top n tokens per topic.

    If a vocabulary is provided, the tokens are returned instead of the types.
    """
    summary = []
    for k in range(topics.shape[1]):
        index = []
        for word in numpy.argsort(topics[:, k])[-n:][::-1]:
            index.append(word)
        summary.append(index)

    if corpus:
        summary = [[corpus.vocabulary[w] for w in topic] for topic in summary]

    return numpy.array(summary)


def _sample_doc(doc, topics, T, alpha, num_iters):
    if not doc.tokens:
        return [], numpy.ones(T) / T

    z = numpy.random.randint(T, size=len(doc.tokens), dtype=int)
    counts = numpy.zeros(T, dtype=int)
    for z_n in z:
        counts[z_n] += 1

    for _ in range(num_iters):
        for n, w_n in enumerate(doc.tokens):
            counts[z[n]] -= 1
            cond = [alpha + counts[t] * topics[w_n.token, t] for t in range(T)]
            z[n] = ankura.util.sample_categorical(cond)
            counts[z[n]] += 1

    return z, counts / counts.sum()


def sampling_assign(corpus_or_doc, topics, alpha=.01, num_iters=10, **kwargs):
    """Predicts topic assignments for a corpus or document.

    Inference is performed using Gibbs sampling with Latent Dirichlet
    Allocation and fixed topics. A symetric Dirichlet prior over the
    document-topic distribution is used.

    By default, the topic distribution or distributions are returned. However,
    if the keyword argument 'return_z' is True, the token level topic
    assignments are returned instead. If the keyword argument 'return_both' is
    True, then both the token level topic assignments and the document topic
    distributions are returned in that order.

    Additionally, if the keyword argument 'metadata_attr' is true, then the
    document topic distributions are added to the metadata of each document
    using the given metadata attribute name. However, this does require that
    the input be a corpus instead of a single document.
    """
    T = topics.shape[1]
    try:
        z, theta = zip(*(_sample_doc(d, topics, T, alpha, num_iters) for d in corpus_or_doc.documents))
    except AttributeError:
        z, theta = _sample_doc(corpus_or_doc, topics, T, alpha, num_iters)

    attr = kwargs.get('metadata_attr')
    if attr:
        for doc, theta_d in zip(corpus_or_doc.documents, theta):
            doc.metadata[attr] = theta_d

    if kwargs.get('return_both'):
        return z, theta
    elif kwargs.get('return_z'):
        return z

    return theta


def variational_assign(data, topics, **kwargs):
    """Predicts topic assignments for a corpus, document or docwords matrix.

    If a corpus or document is given, a sparse docwords matrix is computed. The
    computed or given docwords matrix is then used to compute document topic
    distributions using online variational Bayes with Latent Dirichlet
    Allocation and fixed topics following Hoffman et al., 2010.

    Additionally, if the keyword argument 'metadata_attr' is true, then the
    document topic distributions are added to the metadata of each document
    using the given metadata attribute name. However, this does require that
    the input be a corpus isntead of a single document.
    """
    V, K = topics.shape
    try:
        docwords = ankura.pipeline.build_docwords(data, V)
        out_shape = None
    except AttributeError:
        try:
            docwords = scipy.sparse.lil_matrix((1, V))
            for tl in data.tokens:
                docwords[0, tl.token] += 1
            out_shape = [K]
        except AttributeError:
            docwords = data
            out_shape = None

    lda = sklearn.decomposition.LatentDirichletAllocation(K)
    lda.components_ = topics.T
    lda._init_latent_vars(V)
    theta = lda.transform(docwords)

    attr = kwargs.get('metadata_attr')
    if attr:
        for doc, theta_d in zip(data.documents, theta):
            doc.metadata[attr] = theta_d

    if out_shape:
        return theta.reshape(out_shape)
    return theta


def cross_reference(corpus, attr, doc=None, n=sys.maxsize, threshold=1):
    """Finds the nearest documents by topic similarity.

    The documents of the corpus must include a metadata value giving a vector
    representation of the document. Typically, this is a topic distribution
    obtained with an assign function and a metadata_attr. The vector
    representation is then used to compute distances between documents.

    If a document is given, then a list of references is returned for that
    document. Otherwise, cross references for each document in a corpus are
    given in a dict keyed by the documents. Consequently, the documents of the
    corpus must be hashable.

    The closest n documents will be returned (default=sys.maxsize). Documents
    whose similarity is behond the threshold (default=1) will not be returned.
    A threshold of 1 indicates that no filtering should be done, while a 0
    indicates that only exact matches should be returned.
    """
    def _xrefs(doc):
        doc_theta = doc.metadata[attr]
        dists = [scipy.spatial.distance.cosine(doc_theta, d.metadata[attr])
                 if doc is not d else float('nan')
                 for d in corpus.documents]
        dists = numpy.array(dists)
        return list(corpus.documents[i] for i in dists.argsort()[:n]
                    if dists[i] <= threshold)

    if doc:
        return _xrefs(doc)
    else:
        return {doc: _xrefs(doc) for doc in corpus.documents}


def free_classifier(topics, Q, labels, epsilon=1e-7):
    """There is no free lunch, but this classifier is free"""
    K = len(labels)
    V = Q.shape[0] - K

    # Smooth and column normalize class-topic weights
    A_f = topics[-K:] + epsilon
    A_f /= A_f.sum(axis=0)

    # class_given_word
    Q = Q / Q.sum(axis=1, keepdims=True) # row-normalize Q without original
    Q_L = Q[-K:, :V]

    @functools.wraps(free_classifier)
    def _classifier(doc, theta):
        H = numpy.zeros(V)
        for w_d in doc.tokens:
            H[w_d.token] += 1

        topic_score = A_f.dot(theta)
        topic_score /= topic_score.sum(axis=0)

        word_score = Q_L.dot(H)
        word_score /= word_score.sum(axis=0)

        return labels[numpy.argmax(topic_score + word_score)]
    return _classifier

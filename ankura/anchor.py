"""Functions for anchor selection"""

import collections
import numpy


def gram_schmidt_candidates(corpus, doc_threshold):
    """Generates candidates for finding anchor words using gram_schmidt."""
    counts = collections.Counter()
    for doc in corpus.documents:
        counts.update(set(doc.types))
    return [tid for tid, count in counts.items() if count > doc_threshold]


def gram_schmidt(Q, k, candidates, project_dim=500):
    """Uses stabalized Gram-Schmidt decomposition to find k anchors.

    The candidate anchors are typically choosen using gram_schmidt_candidates.
    """
    # Row-normalize and project Q, preserving the original Q
    Q_orig = Q
    Q = Q / Q.sum(axis=1, keepdims=True)
    if project_dim:
        # Randomly project Q using method from Achlioptas 2001
        R = numpy.random.choice([-1, 0, 0, 0, 0, 1], (Q.shape[1], project_dim))
        Q = numpy.dot(Q, R * numpy.sqrt(3))

    # Setup book keeping
    indices = numpy.zeros(k, dtype=numpy.int)
    basis = numpy.zeros((k-1, Q.shape[1]))

    # Find the farthest point from the origin
    max_dist = 0
    for i in candidates:
        dist = numpy.linalg.norm(Q[i])
        if dist > max_dist:
            max_dist = dist
            indices[0] = i

    # Translate all points to the new origin
    for i in candidates:
        Q[i] = Q[i] - Q[indices[0]]

    # Find the farthest point from origin
    max_dist = 0
    for i in candidates:
        dist = numpy.linalg.norm(Q[i])
        if dist > max_dist:
            max_dist = dist
            indices[1] = i
    basis[0] = Q[indices[1]] / max_dist

    # Stabilized gram-schmidt to finds new anchor words to expand the subspace
    for j in range(1, k - 1):
        # Project all the points onto the basis and find the farthest point
        max_dist = 0
        for i in candidates:
            Q[i] = Q[i] - numpy.dot(Q[i], basis[j-1]) * basis[j - 1]
            dist = numpy.dot(Q[i], Q[i])
            if dist > max_dist:
                max_dist = dist
                indices[j + 1] = i
                basis[j] = Q[i] / numpy.sqrt(numpy.dot(Q[i], Q[i]))

    # Use the original Q to extract anchor vectors using the anchor indices
    return Q_orig[indices, :]

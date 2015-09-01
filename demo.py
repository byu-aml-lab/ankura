"""Runs a demo of the anchor words algorithm"""

import numpy

from ankura import pipeline, anchor, recover

def demo():
    """Runs a demo of the anchors words algorithm"""
    docwords, vocab = pipeline.read_uci('docwords.txt', 'vocab.txt')
    docwords, vocab = pipeline.filter_stopwords(docwords, vocab, 'stop.txt')
    docwords, vocab = pipeline.filter_rarewords(docwords, vocab, 50)

    candidates = anchor.identify_candidates(docwords, 100)
    print len(candidates), 'candidates'

    Q = anchor.construct_Q(docwords)
    print 'Q sum is', Q.sum()

    anchors = anchor.find_anchors(Q, 20, 1000, candidates)
    print anchors
    for ank in anchors:
        print ank, vocab[ank]

    A, _ = recover.nonNegativeRecover(Q, anchors, 'out', 'L2', 1)

    for k in xrange(20):
        topwords = numpy.argsort(A[:, k])[-10:][::-1]
        print vocab[anchors[k]], ':',
        for word in topwords:
            print vocab[word],
        print


if __name__ == '__main__':
    demo()

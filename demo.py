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
    topics = recover.recover_topics(Q, anchors)

    for k in xrange(len(anchors)):
        topwords = numpy.argsort(topics[:, k])[-10:][::-1]
        print vocab[anchors[k]], ':',
        for word in topwords:
            print vocab[word],
        print

if __name__ == '__main__':
    demo()

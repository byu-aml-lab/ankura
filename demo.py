"""Runs a demo of the anchor words algorithm"""

import numpy

import ankura
from ankura import tokenize

def demo():
    """Runs a demo of the anchors words algorithm"""

    data_glob = '/aml/home/jlund3/scratch/data/newsgroups/*/*'
    eng_stop = '/aml/home/jlund3/scratch/data/stopwords/english.txt'
    news_stop = '/aml/home/jlund3/scratch/data/stopwords/newsgroups.txt'

    docwords, vocab = ankura.read_glob(data_glob, tokenize.news)
    docwords, vocab = ankura.filter_stopwords(docwords, vocab, eng_stop)
    docwords, vocab = ankura.filter_stopwords(docwords, vocab, news_stop)
    docwords, vocab = ankura.filter_rarewords(docwords, vocab, 25)

    candidates = ankura.identify_candidates(docwords, 50)

    Q = ankura.construct_Q(docwords)
    print 'Q sum is', Q.sum()

    anchors = ankura.find_anchors(Q, 20, 1000, candidates)
    topics = ankura.recover_topics(Q, anchors)

    for k in xrange(len(anchors)):
        topwords = numpy.argsort(topics[:, k])[-10:][::-1]
        print vocab[anchors[k]], ':',
        for word in topwords:
            print vocab[word],
        print

if __name__ == '__main__':
    demo()

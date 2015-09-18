"""Runs a demo of the anchor words algorithm"""

import time
import datetime
import numpy

import ankura
from ankura import tokenize

NEWS_GLOB = '/local/jlund3/data/newsgroups/*/*'
ENGL_STOP = '/local/jlund3/data/stopwords/english.txt'
NEWS_STOP = '/local/jlund3/data/stopwords/newsgroups.txt'
NEWS_CONSTRAINS = '/local/jlund3/data/constraints/newsgroups.txt'
PIPELINE = [(ankura.read_glob, NEWS_GLOB, tokenize.news),
            (ankura.filter_stopwords, ENGL_STOP),
            (ankura.filter_stopwords, NEWS_STOP),
            (ankura.filter_rarewords, 100),
            (ankura.filter_commonwords, 1500)]

NUM_TOPICS = 20
CAND_THRESH = 500

def demo():
    """Runs a demo of the anchors words algorithm"""
    start = time.time()
    dataset = ankura.run_pipeline(PIPELINE)
    constraints = [line.split() for line in open(NEWS_CONSTRAINS)]
    end = time.time()
    print 'Import took:', datetime.timedelta(seconds=end-start)
    print 'Docwords shape:', dataset.docwords.shape
    print

    start = time.time()
    anchors = ankura.constraint_anchors(dataset, constraints)
    topics = ankura.recover_topics(dataset, anchors)
    end = time.time()
    print 'Topic recovery took:', datetime.timedelta(seconds=end-start)
    print 'Topics:'
    for k in xrange(topics.shape[1]):
        topwords = numpy.argsort(topics[:, k])[-20:][::-1]
        for word in topwords:
            print dataset.vocab[word],
        print

    return topics

if __name__ == '__main__':
    demo()

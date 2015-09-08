"""Runs a demo of the anchor words algorithm"""

import time
import datetime

import ankura

def demo():
    """Runs a demo of the anchors words algorithm"""
    start = time.time()
    pipeline = [(ankura.read_uci, 'docwords.txt', 'vocab.txt'),
                (ankura.filter_stopwords, 'stop.txt'),
                (ankura.filter_rarewords, 25)]
    docwords, vocab = ankura.run_pipeline(pipeline)
    end = time.time()
    print 'Import took:', datetime.timedelta(seconds=end-start)
    print 'Docwords shape:', docwords.shape
    print

    start = time.time()
    candidates = ankura.identify_candidates(docwords, 50)
    Q = ankura.construct_Q(docwords)
    end = time.time()
    print 'Constructing Q took:', datetime.timedelta(seconds=end-start)
    print 'Q sum is', Q.sum()
    print

    start = time.time()
    anchors = ankura.find_anchors(Q, 20, 1000, candidates)
    topics = ankura.recover_topics(Q, anchors)
    end = time.time()
    print 'Topic recovery took:', datetime.timedelta(seconds=end-start)
    print 'Topics:'
    ankura.print_summary(topics, vocab)


if __name__ == '__main__':
    demo()

"""Runs a demo of the anchor words algorithm"""

import time
import datetime

import ankura
from ankura import tokenize


RARE_THRESH = 25
CAND_THRESH = 50
NUM_TOPICS = 20
PROJ_DIMS = 1000
NEWS_GLOB = '/aml/scratch/jlund3/data/newsgroups/*/*'
ENGL_STOP = '/aml/scratch/jlund3/data/stopwords/english.txt'
NEWS_STOP = '/aml/scratch/jlund3/data/stopwords/newsgroups.txt'
PIPELINE = [(ankura.read_glob, NEWS_GLOB, tokenize.news),
            (ankura.filter_stopwords, ENGL_STOP),
            (ankura.filter_stopwords, NEWS_STOP),
            (ankura.filter_rarewords, RARE_THRESH)]

# RARE_THRESH = 25
# CAND_THRESH = 50
# NUM_TOPICS = 20
# PROJ_DIMS = 1000
# PIPELINE = [(ankura.read_uci, 'docwords.txt', 'vocab.txt'),
            # (ankura.filter_stopwords, 'stop.txt'),
            # (ankura.filter_rarewords, RARE_THRESH)]


def demo():
    """Runs a demo of the anchors words algorithm"""
    start = time.time()
    docwords, vocab = ankura.run_pipeline(PIPELINE)
    end = time.time()
    print 'Import took:', datetime.timedelta(seconds=end-start)
    print 'Docwords shape:', docwords.shape
    print
    exit()

    start = time.time()
    candidates = ankura.identify_candidates(docwords, CAND_THRESH)
    Q = ankura.construct_Q(docwords)
    end = time.time()
    print 'Constructing Q took:', datetime.timedelta(seconds=end-start)
    print 'Q sum is', Q.sum()
    print

    start = time.time()
    anchors = ankura.find_anchors(Q, NUM_TOPICS, PROJ_DIMS, candidates)
    topics = ankura.recover_topics(Q, anchors)
    end = time.time()
    print 'Topic recovery took:', datetime.timedelta(seconds=end-start)
    print 'Topics:'
    ankura.print_summary(topics, vocab, prefix=lambda k: vocab[anchors[k]])


if __name__ == '__main__':
    demo()

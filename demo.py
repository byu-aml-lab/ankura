"""Runs a demo of the anchor words algorithm"""

import time
import datetime

import ankura
from ankura import tokenize

RARE_THRESH = 100
COMMON_THRESH = 1500
CAND_THRESH = 500
NUM_TOPICS = 50
PROJ_DIMS = 1000
NEWS_GLOB = '/local/jlund3/data/newsgroups/*/*'
ENGL_STOP = '/local/jlund3/data/stopwords/english.txt'
NEWS_STOP = '/local/jlund3/data/stopwords/newsgroups.txt'
PIPELINE = [(ankura.read_glob, NEWS_GLOB, tokenize.news),
            (ankura.filter_stopwords, ENGL_STOP),
            (ankura.filter_stopwords, NEWS_STOP),
            (ankura.filter_rarewords, RARE_THRESH),
            (ankura.filter_commonwords, COMMON_THRESH)]

# RARE_THRESH = 25
# CAND_THRESH = 50
# NUM_TOPICS = 20
# PROJ_DIMS = 1000
# NIPS_DOCWORDS = 'data/nips.docwords.txt'
# NIPS_VOCAB = 'data/nips.vocab.txt'
# NIPS_STOP = 'data/nips.stop.txt'
# PIPELINE = [(ankura.read_uci, NIPS_DOCWORDS, NIPS_VOCAB),
            # (ankura.filter_stopwords, NIPS_STOP),
            # (ankura.filter_rarewords, RARE_THRESH)]


def demo():
    """Runs a demo of the anchors words algorithm"""
    start = time.time()
    docwords, vocab = ankura.run_pipeline(PIPELINE)
    end = time.time()
    print 'Import took:', datetime.timedelta(seconds=end-start)
    print 'Docwords shape:', docwords.shape
    print

    start = time.time()
    candidates = ankura.identify_candidates(docwords, CAND_THRESH)
    Q = ankura.construct_Q(docwords)
    end = time.time()
    print 'Constructing Q took:', datetime.timedelta(seconds=end-start)
    print 'Q sum is', Q.sum()
    print

    constraints = [line.split() for line in open('/local/jlund3/data/constraints/newsgroups.txt')]

    start = time.time()
    # anchors = ankura.find_anchors(Q, NUM_TOPICS, PROJ_DIMS, candidates)
    anchors = ankura.anchor.constraint_anchors(Q, vocab, constraints)
    topics = ankura.recover_topics(Q, anchors)
    end = time.time()
    print 'Topic recovery took:', datetime.timedelta(seconds=end-start)
    print 'Topics:'
    ankura.print_summary(topics, vocab, num_words=30, prefix=str)


if __name__ == '__main__':
    demo()

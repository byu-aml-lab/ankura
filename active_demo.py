"""Runs a demo of the active learning experiment"""

import time
import datetime
import numpy
import scipy.sparse
import random

import ankura
from ankura import tokenize

import active.evaluate
import active.select
import sampler.slda

SOTU_GLOB = '/net/roi/okuda/state_of_the_union/quarter/*'
ENGL_STOP = '/net/roi/okuda/data/stopwords.txt'
PIPELINE = [(ankura.read_glob, SOTU_GLOB, tokenize.simple),
            (ankura.filter_stopwords, ENGL_STOP),
            (ankura.filter_rarewords, 5),
            (ankura.filter_commonwords, 1500),
            (ankura.filter_smalldocs, 5)]
SOTU_LABELS = '/net/roi/okuda/state_of_the_union/ankura_quarter_timestamps.data'

CAND_SIZE = 500
SEED = 531

NUM_TOPICS = 20
ALPHA = 0.1
BETA = 0.01
VAR = 0.1
NUM_TRAIN = 5
NUM_SAMPLES_TRAIN = 5
TRAIN_BURN = 50
TRAIN_LAG = 50
NUM_SAMPLES_PREDICT = 5
PREDICT_BURN = 10
PREDICT_LAG = 5

START_LABELED = 10
END_LABELED = 100
LABEL_INCREMENT = 10

TEST_SIZE = 200

SELECT_METHOD = active.select.factory['random']

def demo():
    """Runs a demo of active learning simulation with sLDA via sampling"""
    start = time.time()
    rng = random.Random(SEED)
    sampler.slda.set_seed(SEED)
    dataset = ankura.run_pipeline(PIPELINE)
    pre_labels = {}
    with open(SOTU_LABELS) as ifh:
        for line in ifh:
            data = line.strip().split()
            pre_labels[data[0]] = float(data[1])
    labels = []
    for doc_id in range(dataset.num_docs):
        labels.append(pre_labels[dataset.titles[doc_id]])
    # initialize document token ordering
    dataset._pregenerate_doc_tokens()
    end = time.time()
    print 'Import took:', datetime.timedelta(seconds=end-start)
    print

    start = time.time()

    # initialize sets
    shuffled_doc_ids = range(dataset.num_docs)
    rng.shuffle(shuffled_doc_ids)
    test_doc_ids = shuffled_doc_ids[:TEST_SIZE]
    test_labels = []
    test_words = []
    for t in test_doc_ids:
        test_labels.append(labels[t])
        test_words.append(dataset.doc_tokens(t))
    test_labels_mean = numpy.mean(test_labels)
    labeled_doc_ids = shuffled_doc_ids[TEST_SIZE:TEST_SIZE+START_LABELED]
    known_labels = []
    for t in labeled_doc_ids:
        known_labels.append(labels[t])
    unlabeled_doc_ids = set(shuffled_doc_ids[TEST_SIZE+START_LABELED:])

    model = sampler.slda.SamplingSLDA(rng, NUM_TOPICS, ALPHA, BETA, VAR,
            NUM_TRAIN, NUM_SAMPLES_TRAIN, TRAIN_BURN, TRAIN_LAG,
            NUM_SAMPLES_PREDICT, PREDICT_BURN, PREDICT_LAG)

    # learning loop
    model.train(dataset, labeled_doc_ids, known_labels)
    metric = active.evaluate.pR2(model, test_words, test_labels, test_labels_mean)
    print len(labeled_doc_ids), metric
    while len(labeled_doc_ids) < END_LABELED and len(unlabeled_doc_ids) > 0:
        candidates = active.select.reservoir(list(unlabeled_doc_ids),
                rng, CAND_SIZE)
        chosen = SELECT_METHOD(candidates, model, rng, LABEL_INCREMENT)
        for c in chosen:
            known_labels.append(labels[c])
            labeled_doc_ids.append(c)
            unlabeled_doc_ids.remove(c)
        model.train(dataset, labeled_doc_ids, known_labels, True)
        metric = active.evaluate.pR2(model, test_words, test_labels, test_labels_mean)
        print len(labeled_doc_ids), metric
    model.cleanup()
    end = time.time()
    print
    print 'Simulation took:', datetime.timedelta(seconds=end-start)
    print


if __name__ == '__main__':
    demo()

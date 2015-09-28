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
    labels = {}
    with open(SOTU_LABELS) as ifh:
        for line in ifh:
            data = line.strip().split()
            labels[data[0]] = float(data[1])
    # expand dataset into word vectors
    corpus = {}
    for t in dataset.titles:
        corpus[t] = []
    indexTitleMap = {}
    for (i, title) in enumerate(dataset.titles):
        indexTitleMap[i] = title
    (rows, cols, vals) = scipy.sparse.find(dataset.M)
    for (i, col) in enumerate(cols):
        for _ in range(int(vals[i])):
            corpus[indexTitleMap[col]].append(dataset.vocab[int(rows[i])])
    # make sure that word types aren't all clumped up together
    for title in corpus:
        rng.shuffle(corpus[title])
    with open('corpus.txt', 'w') as ofh:
        for title in corpus:
            ofh.write(title)
            ofh.write('\t')
            ofh.write(' '.join(corpus[title]))
            ofh.write('\n')
    end = time.time()
    print 'Import took:', datetime.timedelta(seconds=end-start)
    print

    start = time.time()

    # initialize sets
    shuffled_titles = list(dataset.titles)
    rng.shuffle(shuffled_titles)
    test_titles = shuffled_titles[:TEST_SIZE]
    test_labels = []
    test_words = []
    for t in test_titles:
        test_labels.append(labels[t])
        test_words.append(corpus[t])
    test_labels_mean = numpy.mean(test_labels)
    labeled_titles = shuffled_titles[TEST_SIZE:TEST_SIZE+START_LABELED]
    known_labels = []
    for t in labeled_titles:
        known_labels.append(labels[t])
    unlabeled_titles = set(shuffled_titles[TEST_SIZE+START_LABELED:])

    model = sampler.slda.SamplingSLDA(rng, NUM_TOPICS, ALPHA, BETA, VAR,
            NUM_TRAIN, NUM_SAMPLES_TRAIN, TRAIN_BURN, TRAIN_LAG,
            NUM_SAMPLES_PREDICT, PREDICT_BURN, PREDICT_LAG)

    # learning loop
    model.train(corpus, labeled_titles, known_labels)
    metric = active.evaluate.pR2(model, test_words, test_labels, test_labels_mean)
    print len(labeled_titles), metric
    while len(labeled_titles) < END_LABELED and len(unlabeled_titles) > 0:
        candidates = active.select.reservoir(list(unlabeled_titles),
                rng, CAND_SIZE)
        chosen = SELECT_METHOD(candidates, model, rng, LABEL_INCREMENT)
        for c in chosen:
            known_labels.append(labels[c])
            labeled_titles.append(c)
            unlabeled_titles.remove(c)
        model.train(corpus, labeled_titles, known_labels, True)
        metric = active.evaluate.pR2(model, test_words, test_labels, test_labels_mean)
        print len(labeled_titles), metric
    model.cleanup()
    with open('known.txt', 'w') as ofh:
        for t, l in zip(labeled_titles, known_labels):
            ofh.write(t)
            ofh.write('\t')
            ofh.write(str(l))
            ofh.write('\n')
    end = time.time()
    print
    print 'Simulation took:', datetime.timedelta(seconds=end-start)
    print


if __name__ == '__main__':
    demo()

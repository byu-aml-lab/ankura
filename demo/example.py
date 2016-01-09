"""Runs a demo of the anchor words algorithm

This is really a proof of concept and is ugly as sin. You'll need to edit the
file paths to point to your clone of github.com:jlund3/data for this to work.
"""

from __future__ import print_function

import os
import numpy

import ankura

@ankura.util.memoize
@ankura.util.pickle_cache('newsgroups.pickle')
def get_newsgroups():
    """Retrieves the 20 newsgroups dataset"""
    news_glob = '/local/cojoco/git/jeffData/newsgroups/*/*'
    engl_stop = '/local/cojoco/git/jeffData/stopwords/english.txt'
    news_stop = '/local/cojoco/git/jeffData/stopwords/newsgroups.txt'
    name_stop = '/local/cojoco/git/jeffData/stopwords/malenames.txt'
    pipeline = [(ankura.read_glob, news_glob, ankura.tokenize.news),
                (ankura.filter_stopwords, engl_stop),
                (ankura.filter_stopwords, news_stop),
                (ankura.combine_words, name_stop, '<name>'),
                (ankura.filter_rarewords, 100),
                (ankura.filter_commonwords, 1500)]
    dataset = ankura.run_pipeline(pipeline)
    return dataset


@ankura.util.memoize
@ankura.util.pickle_cache('anchors-default.pickle')
def default_anchors():
    """Retrieves default anchors for newsgroups using Gram-Schmidt"""
    return ankura.gramschmidt_anchors(get_newsgroups(), 20, 500)


@ankura.util.memoize
def constraint_anchors(dataset):
    """Retrieves anchors for newsgroups based on information gain"""
    anchors = open('/local/jlund3/data/constraints/newsgroups.txt').readlines()
    anchors = [line.strip().split() for line in anchors]
    anchors = [[w for w in a if w in dataset.vocab] for a in anchors]
    anchors = [[dataset.vocab.index(w) for w in a] for a in anchors]
    return ankura.util.tuplize(anchors)


@ankura.util.memoize
def get_topics(dataset, anchors):
    """Gets the topics for 20 newsgroups given a set of anchors"""
    return ankura.recover_topics(dataset, anchors)


def main():
    """Runs the example code"""
    dataset = get_newsgroups()

    anchors = default_anchors()
    # anchors = constraint_anchors(dataset)

    topics = get_topics(dataset, anchors)

    for k in range(topics.shape[1]):
        print(k, end=': ')
        for word in numpy.argsort(topics[:, k])[-20:][::-1]:
            print(dataset.vocab[word], end=' ')
        print()

    tdataset = ankura.topic_transform(topics, dataset)

    titles = [os.path.dirname(t) for t in tdataset.titles]
    naive = ankura.measure.NaiveBayes(tdataset, titles)
    print(naive.validate(tdataset, titles))

if __name__ == '__main__':
    main()

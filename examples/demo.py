"""A demo of ankura functionality"""

import numpy
import os

import ankura
from ankura import measure


@ankura.util.memoize
@ankura.util.pickle_cache('newsgroups-dataset.pickle')
def get_dataset():
    """Retrieves the 20 newsgroups dataset"""
    news_glob = '/local/jlund3/data/newsgroups/*/*'
    engl_stop = '/local/jlund3/data/stopwords/english.txt'
    news_stop = '/local/jlund3/data/stopwords/newsgroups.txt'
    name_stop = '/local/jlund3/data/stopwords/malenames.txt'
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
def get_anchors():
    """Retrieves default anchors for newsgroups using Gram-Schmidt"""
    return ankura.gramschmidt_anchors(get_dataset(), 20, 500)


def get_topics(dataset, anchors):
    """Wraps recover topics with memoize"""
    return ankura.recover_topics(dataset, anchors)


def print_summary(dataset, topics):
    """Prints a summary of the given topics"""
    for k in range(topics.shape[1]):
        summary = []
        for word in numpy.argsort(topics[:, k])[-10:][::-1]:
            summary.append(dataset.vocab[word])
        print(' '.join(summary))


def demo():
    """Runs the demo"""
    dataset = get_dataset()
    anchors = get_anchors()
    topics = get_topics(dataset, anchors)
    print_summary(dataset, topics)

    trans = ankura.topic_combine(topics, dataset)
    labels = [os.path.dirname(title) for title in trans.titles]
    naive = measure.NaiveBayes(trans, labels)
    accuracy = naive.validate(trans, labels)
    print('Accuracy:', accuracy)

if __name__ == '__main__':
    demo()

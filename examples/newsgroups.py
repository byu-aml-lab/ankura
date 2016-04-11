"""A demo of ankura functionality"""

from __future__ import print_function

import os
import numpy

import ankura

@ankura.util.memoize
@ankura.util.pickle_cache('newsgroups.pickle')
def get_newsgroups():
    """Retrieves the 20 newsgroups dataset"""
    datadir = '/local/jlund3/data/'

    news_glob = datadir + 'newsgroups-dedup/*/*'
    engl_stop = datadir + 'stopwords/english.txt'
    news_stop = datadir + 'stopwords/newsgroups.txt'
    name_stop = datadir + 'stopwords/malenames.txt'

    dataset = ankura.read_glob(news_glob,
                               tokenizer=ankura.tokenize.news,
                               labeler=ankura.label.title_dirname)
    dataset = ankura.filter_stopwords(dataset, engl_stop)
    dataset = ankura.filter_stopwords(dataset, news_stop)
    dataset = ankura.combine_words(dataset, name_stop, '<name>')
    dataset = ankura.filter_rarewords(dataset, 100)
    dataset = ankura.filter_commonwords(dataset, 1500)

    for doc, title in enumerate(dataset.titles):
        title = title[len(datadir + 'newsgroups-dedup/'):]
        dataset.titles[doc] = title
        outpath = os.path.join(datadir, 'newsgroups-processed', title)
        try:
            os.makedirs(os.path.dirname(outpath))
        except FileExistsError:
            pass

        with open(outpath, 'w') as outfile:
            tokens = [dataset.vocab[v] for v in dataset.doc_tokens(doc)]
            print(' '.join(tokens), file=outfile)

    return dataset


def get_title_anchors(dataset):
    """Retrieves anchors constructed from the newsgroup titles"""
    anchor_tokens = [
        ['computer', 'graphics'],
        ['computer', 'operating', 'system', 'microsoft', 'windows'],
        ['computer', 'ibm', 'pc', 'hardware'],
        ['computer', 'mac', 'hardware'],
        ['computer', 'windows'],
        ['auto'],
        ['recreation', 'motorcycle'],
        ['recreation', 'sport', 'baseball'],
        ['recreation', 'sport', 'hockey'],
        ['talk', 'politics'],
        ['talk', 'politics', 'guns'],
        ['talk', 'politics', 'middle', 'east'],
        ['science', 'cryptography'],
        ['science', 'electronics'],
        ['science', 'medicine'],
        ['science', 'space'],
        ['talk', 'religion'],
        ['alternative', 'atheism'],
        ['social', 'religion', 'christian'],
    ]
    return ankura.multiword_anchors(dataset, anchor_tokens)


def demo():
    """Runs the demo"""
    dataset = get_newsgroups()

    def run(name, anchors):
        topics = ankura.recover_topics(dataset, anchors)
        features = ankura.topic_combine(topics, dataset)
        train, test = ankura.pipeline.train_test_split(features, .9)

        vw_contingency = ankura.measure.vowpal_contingency(train, test, 'dirname')
        print(name, 'accuracy:', ankura.measure.vowpal_accuracy(train, test, 'dirname'))
        print(name, 'f-Measure:', vw_contingency.fmeasure())
        print(name, 'ari:', vw_contingency.ari())
        print(name, 'rand:', vw_contingency.rand())
        print(name, 'vi:', vw_contingency.vi())

        coherence = []
        for topic in ankura.topic.topic_summary_indices(topics, dataset, 10):
            coherence.append(ankura.measure.topic_coherence(topic, dataset))
        print(name, 'coherence-10:', numpy.mean(coherence))

        coherence = []
        for topic in ankura.topic.topic_summary_indices(topics, dataset, 15):
            coherence.append(ankura.measure.topic_coherence(topic, dataset))
        print(name, 'coherence-15:', numpy.mean(coherence))

        coherence = []
        for topic in ankura.topic.topic_summary_indices(topics, dataset, 20):
            coherence.append(ankura.measure.topic_coherence(topic, dataset))
        print(name, 'coherence-20:', numpy.mean(coherence))

    run('default', ankura.gramschmidt_anchors(get_newsgroups(), 20, 500))
    run('title', get_title_anchors(dataset))


if __name__ == '__main__':
    demo()

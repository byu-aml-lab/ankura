"""A demo of ankura functionality"""

from __future__ import print_function

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

    return dataset


def demo():
    """Runs the newsgroups demo"""
    dataset = get_newsgroups()
    anchors = ankura.gramschmidt_anchors(dataset, 20, 500)
    topics = ankura.recover_topics(dataset, anchors)

    for topic in ankura.topic.topic_summary_tokens(topics, dataset, 20):
        print(' '.join(topic))


if __name__ == '__main__':
    demo()

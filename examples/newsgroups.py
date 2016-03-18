"""A demo of ankura functionality"""

import os

import ankura

@ankura.util.memoize
@ankura.util.pickle_cache('newsgroups.pickle')
def get_newsgroups():
    """Retrieves the 20 newsgroups dataset"""
    news_glob = '/local/jlund3/data/newsgroups/*/*'
    engl_stop = '/local/jlund3/data/stopwords/english.txt'
    news_stop = '/local/jlund3/data/stopwords/newsgroups.txt'
    name_stop = '/local/jlund3/data/stopwords/malenames.txt'

    dataset = ankura.read_glob(news_glob, tokenizer=ankura.tokenize.news)
    dataset = ankura.filter_stopwords(dataset, engl_stop)
    dataset = ankura.filter_stopwords(dataset, news_stop)
    dataset = ankura.combine_words(dataset, name_stop, '<name>')
    dataset = ankura.filter_rarewords(dataset, 100)
    dataset = ankura.filter_commonwords(dataset, 1500)

    return dataset


def demo():
    """Runs the demo"""
    dataset = get_newsgroups()
    anchors = ankura.gramschmidt_anchors(get_newsgroups(), 20, 500)
    topics = ankura.recover_topics(dataset, anchors)
    for topic in ankura.topic.topic_summary(topics, dataset, n):
        print(' '.join(topic))



if __name__ == '__main__':
    demo()

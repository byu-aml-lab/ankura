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

    dataset = ankura.read_glob(news_glob,
                               tokenizer=ankura.tokenize.news,
                               labeler=ankura.label.title_dirname)
    dataset = ankura.filter_stopwords(dataset, engl_stop)
    dataset = ankura.filter_stopwords(dataset, news_stop)
    dataset = ankura.combine_words(dataset, name_stop, '<name>')
    dataset = ankura.filter_rarewords(dataset, 100)
    dataset = ankura.filter_commonwords(dataset, 1500)

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
        print(name)
        topics = ankura.recover_topics(dataset, anchors)
        for topic in ankura.topic.topic_summary_tokens(topics, dataset, 15):
            print(' '.join(topic))

        transform = ankura.topic_transform(topics, dataset)
        train, test = ankura.pipeline.train_test_split(transform, .9)
        naive = ankura.measure.NaiveBayes(train, dirname)
        accuracy = naive.validate(test)
        print('accuracy:', accuracy)

    run('default', ankura.gramschmidt_anchors(get_newsgroups(), 20, 500))
    run('title', get_title_anchors(dataset))


if __name__ == '__main__':
    demo()

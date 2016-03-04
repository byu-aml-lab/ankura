"""A demo of ankura functionality"""

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


def get_gramschmidt_anchors():
    """Retrieves default anchors for newsgroups using Gram-Schmidt"""
    return ankura.gramschmidt_anchors(get_dataset(), 20, 500)


def get_title_anchors(dataset):
    """Retrieves anchors constructed from the newsgroup titles"""
    anchor_tokens = [
        ['computer', 'graphics'],
        ['computer', 'microsoft', 'windows'],
        ['computer', 'ibm', 'pc', 'hardware'],
        ['computer', 'mac', 'hardware'],
        ['computer', 'windows'],
        ['auto'],
        ['motorcycle'],
        ['baseball'],
        ['hockey'],
        ['talk', 'politics'],
        ['talk', 'politics', 'guns'],
        ['talk', 'politics', 'middle', 'east'],
        ['science', 'cryptography'],
        ['science', 'electronics'],
        ['science', 'medicine'],
        ['science', 'space'],
        ['talk', 'religion'],
        ['alternative', 'religion', 'atheism'],
        ['social', 'religion', 'christian'],
    ]
    return ankura.multiword_anchors(dataset, anchor_tokens)


def get_oracular_anchors(dataset, combiner=ankura.vector_average):
    """Retrieves anchors created by an expert user"""
    anchor_tokens = [
        ['graphics', 'card', 'video'],
        ['windows', 'microsoft', 'nt'],
        ['ibm', 'hardware'],
        ['mac', 'hardware', 'apple'],
        ['windows'],
        ['car', 'cars', 'auto'],
        ['motorcycle', 'bike', 'bikes'],
        ['baseball', 'ball', 'base'],
        ['hockey', 'stick'],
        ['politics', 'government', 'president'],
        ['guns', 'gun', 'control'],
        ['israel', 'palestine', 'palestinian', 'jew', 'israeli'],
        ['armenia', 'genocide', 'turkey'],
        ['cryptography', 'crypto', 'key'],
        ['electronic', 'electronics'],
        ['medicine', 'disease'],
        ['space', 'program', 'nasa', 'shuttle'],
        ['religion'],
        ['atheism', 'proof', 'prove', 'science'],
        ['christian', 'god', 'jesus'],
        ['sale', 'price', 'sell', 'condition', 'shipping'],
    ]
    return ankura.multiword_anchors(dataset, anchor_tokens, combiner)


def print_summary(topics, dataset, n):
    """Prints a summary of the given topics"""
    for topic in ankura.topic.topic_summary(topics, dataset, n):
        print(' '.join(topic))


def demo():
    """Runs the demo"""
    dataset = get_dataset()
    # anchors = get_gramschmidt_anchors()
    # anchors = get_title_anchors(dataset)
    anchors = get_oracular_anchors(dataset, combiner=ankura.vector_min)
    topics = ankura.recover_topics(dataset, anchors)
    print_summary(topics, dataset, 20)

    # trans = ankura.topic_transform(topics, dataset)
    trans = ankura.topic_combine(topics, dataset)
    labels = [os.path.dirname(title) for title in trans.titles]
    naive = measure.NaiveBayes(trans, labels)
    accuracy = naive.validate(trans, labels)
    print('Accuracy:', accuracy)



if __name__ == '__main__':
    demo()

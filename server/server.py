#!/usr/bin/env python3

"""Runs a user interface for the interactive anchor words algorithm"""

import flask
import functools
import json
import numpy
import os
import random
import re
import sys
import argparse
from datetime import datetime

import ankura
from ankura import label

app = flask.Flask(__name__, static_url_path='')


def find_display_candidates(dataset, curse_stop):
    """Adds display candidates to a dataset"""
    # get display candidates
    candidates = filter(lambda d: len(dataset.metadata[d]['text']) < 3000,
                        range(dataset.num_docs))
    candidates = list(candidates)
    dataset.display_candidates = candidates

    # clean up display candidates text metadata for our byu audience
    for curse in open(curse_stop):
        regex = re.compile(r'\b({})\b'.format(curse), re.M | re.I)
        replace = '*' * len(curse)
        for doc in candidates:
            text = dataset.doc_metadata(doc, 'text')
            text = regex.sub(replace, text)
            dataset.metadata[doc]['text'] = text

    return dataset


@ankura.util.memoize
@ankura.util.pickle_cache('newsgroups-dataset.pickle')
def get_newsgroups():
    """Retrieves the 20 newsgroups dataset"""
    news_glob =  os.path.join(args.data_prefix, 'newsgroups/*/*')
    engl_stop =  os.path.join(args.data_prefix, 'stopwords/english.txt')
    news_stop =  os.path.join(args.data_prefix, 'stopwords/newsgroups.txt')
    name_stop =  os.path.join(args.data_prefix, 'stopwords/malenames.txt')
    curse_stop = os.path.join(args.data_prefix, 'stopwords/profanity.txt')
    news_text = functools.partial(label.text, formatter=label.news_formatter)

    dataset = ankura.read_glob(news_glob,
                               tokenizer=ankura.tokenize.news,
                               labeler=[news_text, label.title_dirname])
    dataset = ankura.filter_stopwords(dataset, engl_stop)
    dataset = ankura.filter_stopwords(dataset, news_stop)
    dataset = ankura.combine_words(dataset, name_stop, '<name>')
    dataset = ankura.combine_words(dataset, curse_stop, '<profanity>')
    dataset = ankura.filter_rarewords(dataset, 100)
    dataset = ankura.filter_commonwords(dataset, 1500)
    dataset = find_display_candidates(dataset, curse_stop)

    return dataset


@ankura.util.memoize
@ankura.util.pickle_cache('newsgroups-anchors.pickle')
def newsgroup_anchors():
    """Retrieves default anchors for newsgroups using Gram-Schmidt"""
    dataset = get_newsgroups()
    anchors, indices = ankura.gramschmidt_anchors(dataset, 20, 500,
                                                  return_indices=True)
    anchor_tokens = [[dataset.vocab[index]] for index in indices]
    return anchor_tokens, anchors


@ankura.util.memoize
@ankura.util.pickle_cache('amazon-dataset.pickle')
def get_amazon():
    """Retrieves the amazon dataset"""
    text_path = os.path.join(args.data_prefix, 'amazon', 'amazon.txt')
    engl_stop =  os.path.join(args.data_prefix, 'stopwords/english.txt')
    curse_stop = os.path.join(args.data_prefix, 'stopwords/profanity.txt')

    dataset = ankura.read_file(text_path, labeler=label.text)
    dataset = ankura.filter_stopwords(dataset, engl_stop)
    dataset = ankura.combine_words(dataset, curse_stop, '<profanity>')
    dataset = ankura.filter_rarewords(dataset, 150)
    dataset = ankura.filter_commonwords(dataset, 3000)
    dataset = find_display_candidates(dataset, curse_stop)

    return dataset


@ankura.util.memoize
@ankura.util.pickle_cache('amazon-anchors.pickle')
def amazon_anchors():
    """Retrieves default anchors for amazon using Gram-Schmidt"""
    dataset = get_amazon()
    anchors, indices = ankura.gramschmidt_anchors(dataset, 30, 500,
                                                  return_indices=True)
    anchor_tokens = [[dataset.vocab[index]] for index in indices]
    return anchor_tokens, anchors


datasets = {'newsgroups': (get_newsgroups, newsgroup_anchors),
            'amazon': (get_amazon, amazon_anchors)}

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_prefix',
                    default='/local/jlund3/data',
                    help='The directory where data repo lives')
parser.add_argument('-u', '--user_data',
                    default='user_data',
                    help='The directory where user data is saved')
parser.add_argument('-s', '--single_anchors',
                    action="store_true",
                    help='Enables single-anchors mode')
parser.add_argument('-p', '--port',
                    type=int, default=5000,
                    help='Port which should be used')
parser.add_argument('--docs-per-topic',
                    default=5, type=int,
                    help='Number of documents per topic to display')
parser.add_argument('--dataset',
                    choices=list(datasets.keys()),
                    default='newsgroups',
                    help='The name of the dataset to run on')
args = parser.parse_args()
args.get_dataset, args.default_anchors = datasets[args.dataset]


@ankura.util.memoize
def user_anchors(anchor_tokens):
    """Computes multiword anchors from user specified anchor tokens"""
    return ankura.multiword_anchors(args.get_dataset(), anchor_tokens)


@app.route('/')
def serve_itm():
    """Serves the Interactive Topic Modeling UI"""
    return app.send_static_file('index.html')


@app.route('/finished', methods=['GET', 'POST'])
def save_user_data():
    """Receives and saves user data when done button is clicked in the ITM UI"""
    flask.request.get_data()
    input_json = flask.request.get_json(force=True)
    with ankura.util.open_unique(dirname=args.user_data) as data_file:
        json.dump(input_json, data_file)
    return 'OK'


@app.route('/vocab')
def get_vocab():
    """Returns all valid vocabulary words in the dataset"""
    return flask.jsonify(vocab=args.get_dataset().vocab)


@ankura.util.memoize
def topic_inference(raw_anchors):
    """Returns infered topic info from raw anchors"""
    dataset = args.get_dataset()

    if raw_anchors is None:
        anchor_tokens, anchors = args.default_anchors()
    else:
        anchor_tokens = ankura.util.tuplize(json.loads(raw_anchors))
        anchors = user_anchors(anchor_tokens)

    topics = ankura.recover_topics(dataset, anchors, epsilon=1e-6)
    topic_summary = ankura.topic.topic_summary_tokens(topics, dataset, n=15)

    return topics, topic_summary, anchor_tokens


@app.route('/topics')
def topic_request():
    """Performs a topic request using anchors from the query string"""
    # get the anchors (both tokens and vector) from the request
    raw_anchors = flask.request.args.get('anchors')
    topics, topic_summary, anchor_tokens = topic_inference(raw_anchors)

    # get sample documents for each topic
    dataset = args.get_dataset()
    random.shuffle(dataset.display_candidates)
    docdata = [[] for _ in range(topics.shape[1])]
    for doc in dataset.display_candidates:
        doc_tokens = dataset.doc_tokens(doc)
        counts, _ = ankura.topic.predict_topics(topics, doc_tokens)
        max_topic = int(numpy.argmax(counts))
        if len(docdata[max_topic]) < args.docs_per_topic:
            docdata[max_topic].append(dataset.doc_metadata(doc, 'text'))
        if min(len(docs) for docs in docdata) == args.docs_per_topic:
            break
    accuracy = 0.752283200000001

    return flask.jsonify(anchors=anchor_tokens,
                         topics=topic_summary,
                         examples=docdata,
                         single_anchors=args.single_anchors,
                         accuracy=accuracy)

if __name__ == '__main__':
    # call these to trigger pickle_cache
    args.get_dataset()
    args.default_anchors()

    # start the server, with the data already cached
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=args.port)

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

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_prefix',
                    default='/local/jlund3/data',
                    help='The directory where newsgroups lives')
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
args = parser.parse_args()


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
    labeler = label.aggregate(news_text, label.title_dirname)

    dataset = ankura.read_glob(news_glob, tokenizer=ankura.tokenize.news,
                               labeler=labeler)
    dataset = ankura.filter_stopwords(dataset, engl_stop)
    dataset = ankura.filter_stopwords(dataset, news_stop)
    dataset = ankura.combine_words(dataset, name_stop, '<name>')
    dataset = ankura.combine_words(dataset, curse_stop, '<profanity>')
    dataset = ankura.filter_rarewords(dataset, 100)
    dataset = ankura.filter_commonwords(dataset, 1500)

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
@ankura.util.pickle_cache('newsgroups-anchors.pickle')
def default_anchors():
    """Retrieves default anchors for newsgroups using Gram-Schmidt"""
    dataset = get_newsgroups()
    anchors, indices = ankura.gramschmidt_anchors(dataset, 20, 500,
                                                  return_indices=True)
    anchor_tokens = [[dataset.vocab[index]] for index in indices]
    return anchor_tokens, anchors


@ankura.util.memoize
def user_anchors(anchor_tokens):
    """Computes multiword anchors from user specified anchor tokens"""
    return ankura.multiword_anchors(get_newsgroups(), anchor_tokens)


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
    return flask.jsonify(vocab=get_newsgroups().vocab)


@ankura.util.memoize
def topic_inference(raw_anchors):
    """Returns infered topic info from raw anchors"""
    dataset = get_newsgroups()

    if raw_anchors is None:
        anchor_tokens, anchors = default_anchors()
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
    dataset = get_newsgroups()
    random.shuffle(dataset.display_candidates)
    docdata = [[] for _ in range(topics.shape[1])]
    for doc in dataset.display_candidates:
        doc_tokens = dataset.doc_tokens(doc)
        counts, doc_topics = ankura.topic.predict_topics(topics, doc_tokens)
        max_topic = int(numpy.argmax(counts))
        if len(docdata[max_topic]) < args.docs_per_topic:
            docdata[max_topic].append(dataset.doc_metadata(doc, 'text'))
        if min(len(docs) for docs in docdata) == args.docs_per_topic:
            break

    return flask.jsonify(anchors=anchor_tokens,
                         topics=topic_summary,
                         examples=docdata,
                         single_anchors=args.single_anchors)


if __name__ == '__main__':
    # call these to trigger pickle_cache
    get_newsgroups()
    default_anchors()

    # start the server, with the data already cached
    app.run(debug=True, host='0.0.0.0', port=args.port)

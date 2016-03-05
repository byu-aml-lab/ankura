#!/usr/bin/env python3

"""Runs a user interface for the interactive anchor words algorithm"""

import json
import os
import tempfile

import flask

import ankura
from ankura import label

app = flask.Flask(__name__, static_url_path='')


@ankura.util.memoize
@ankura.util.pickle_cache('newsgroups.pickle')
def get_newsgroups():
    """Retrieves the 20 newsgroups dataset"""
    news_glob = '/local/jlund3/data/newsgroups/*/*'
    engl_stop = '/local/jlund3/data/stopwords/english.txt'
    news_stop = '/local/jlund3/data/stopwords/newsgroups.txt'
    name_stop = '/local/jlund3/data/stopwords/malenames.txt'
    labeler = label.aggregate(label.text, label.title_dirname)

    dataset = ankura.read_glob(news_glob, tokenizer=ankura.tokenize.news,
                                          labeler=labeler)
    dataset = ankura.filter_stopwords(dataset, engl_stop)
    dataset = ankura.filter_stopwords(dataset, news_stop)
    dataset = ankura.combine_words(dataset, name_stop, '<name>')
    dataset = ankura.filter_rarewords(dataset, 100)
    dataset = ankura.filter_commonwords(dataset, 1500)

    return dataset


@ankura.util.memoize
@ankura.util.pickle_cache('anchors-default.pickle')
def default_anchors():
    """Retrieves default anchors for newsgroups using Gram-Schmidt"""
    dataset = get_newsgroups()
    anchors, anchor_indices = ankura.gramschmidt_anchors(dataset,
                                                         20,
                                                         500,
                                                         return_indices=True)
    anchor_tokens = [[dataset.vocab[index]] for index in anchor_indices]
    return anchor_tokens, anchors


@ankura.util.memoize
def user_anchors(anchor_tokens):
    """Computes multiword anchors from user specified anchor tokens"""
    return ankura.multiword_anchors(get_newsgroups(), anchor_tokens)


@app.route('/topics')
def topic_request():
    """Performs a topic request using anchors from the query string"""
    dataset = get_newsgroups()
    raw_anchors = flask.request.args.get('anchors')

    if raw_anchors is None:
        anchor_tokens, anchors = default_anchors()
    else:
        anchor_tokens = ankura.util.tuplize(json.loads(raw_anchors))
        anchors = user_anchors(anchor_tokens)

    topics = ankura.recover_topics(dataset, anchors)
    topic_summary = ankura.topic.topic_summary(topics, dataset, n=15)

    return flask.jsonify(topics=topic_summary, anchors=anchor_tokens)


@app.route('/')
def serve_itm():
    """Serves the Interactive Topic Modeling UI"""
    return app.send_static_file('index.html')


@app.route('/finished', methods=['GET', 'POST'])
def get_user_data():
    """Receives and saves user data when done button is clicked in the ITM UI"""
    flask.request.get_data()
    input_json = flask.request.get_json(force=True)
    user_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/userData"
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir)
    with tempfile.NamedTemporaryFile(mode='w', prefix="itmUserData", dir=os.path.dirname(os.path.realpath(__file__)) + "/userData", delete=False) as dataFile:
        json.dump(input_json, dataFile, sort_keys=True, indent=2, ensure_ascii=False)
    return 'OK'


@app.route('/vocab')
def get_vocab():
    """Returns all valid vocabulary words in the dataset"""
    return flask.jsonify(vocab=get_newsgroups().vocab)


@app.route('/cooccurrences')
def get_cooccurrences():
    """Returns the cooccurrences matrix from the dataset"""
    dataset = get_newsgroups()
    return flask.jsonify(cooccurrences=dataset.Q.tolist())


if __name__ == '__main__':
    default_anchors()
    app.run(debug=True, host='0.0.0.0')

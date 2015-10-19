#!/usr/bin/env python
"""Runs a demo of the anchor words algorithm

This is really a proof of concept and is as ugly as sin.

Note that if you want to run this, you'll have to update the file paths in the
get_newsgroups function to be something useful to you. Those paths currently
point at a clone of github.com/jlund3/data, so you may want to clone my data
repo as well.
"""

import os
import pickle
import flask
import json
import numpy

import ankura

app = flask.Flask(__name__, static_url_path='')


def pickle_cache(pickle_path):
    """Decorator to cache a function call to disk"""
    def _cache(data_func):
        def _load_data():
            if os.path.exists(pickle_path):
                print ' * Reading cached data from disk'
                return pickle.load(open(pickle_path))
            else:
                data = data_func()
                print ' * Caching data to disk'
                pickle.dump(data, open(pickle_path, 'w'))
                return data
        return _load_data
    return _cache


class memoize(object): # pylint: disable=invalid-name
    """Decorator to memoize a function"""

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

@memoize
@pickle_cache('newsgroups.pickle')
def get_newsgroups():
    """Retrieves the 20 newsgroups dataset"""
    news_glob = '/local/cojoco/git/jeffData/newsgroups/*/*'
    engl_stop = '/local/cojoco/git/jeffData/stopwords/english.txt'
    news_stop = '/local/cojoco/git/jeffData/stopwords/newsgroups.txt'
    pipeline = [(ankura.read_glob, news_glob, ankura.tokenize.news),
                (ankura.filter_stopwords, engl_stop),
                (ankura.filter_stopwords, news_stop),
                (ankura.filter_rarewords, 100),
                (ankura.filter_commonwords, 1500)]
    print ' * Import pipeline running'
    dataset = ankura.run_pipeline(pipeline)
    print ' * Pipeline complete'
    return dataset


@memoize
def get_topics(dataset, anchors):
    """Gets the topics for 20 newsgroups given a set of anchors"""
    return ankura.recover_topics(dataset, anchors)


def reindex_token(dataset, token):
    """Converts any string tokens to index of the token"""
    return token if isinstance(token, int) else dataset.vocab.index(token)

def reindex_anchor(dataset, anchor):
    """Converts any tokens in an anchor to the index of token"""
    return tuple(reindex_token(dataset, token) for token in anchor)


def reindex_anchors(dataset, anchors):
    """Converts any tokens in a set of anchors to the index of token"""
    return tuple(reindex_anchor(dataset, anchor) for anchor in anchors)


def tokenify_anchor(dataset, anchor):
    """Converts any token indexes in an anchors to tokens"""
    return tuple(dataset.vocab[index] for index in anchor)


def tokenify_anchors(dataset, anchors):
    """Converts any token indexes in a set of anchors to tokens"""
    return tuple(tokenify_anchor(dataset, anchor) for anchor in anchors)


@app.route('/topics')
def topic_request():
    """Performs a topic request using anchors from the query string"""
    dataset = get_newsgroups()
    raw_anchors = flask.request.args.get('anchors')

    if raw_anchors is None:
        anchors = ankura.gramschmidt_anchors(dataset, 20, 500)
    else:
        anchors = json.loads(raw_anchors)
    anchors = reindex_anchors(dataset, anchors)
    topics = get_topics(dataset, anchors)

    topic_summary = []
    for k in xrange(topics.shape[1]):
        summary = []
        for word in numpy.argsort(topics[:, k])[-10:][::-1]:
            summary.append(dataset.vocab[word])
        topic_summary.append(summary)

    if raw_anchors is None:
        return flask.jsonify(
            topics=topic_summary,
            anchors=tokenify_anchors(dataset, anchors)
        )
    else:
        return flask.jsonify(topics=topic_summary)


@app.route('/demo')
def root():
    """Serves up the single page app which demos interactive topics"""
    return flask.send_from_directory('.', 'demo.html')

@app.route('/test1')
def serve_test1():
    """Serves up the first test for making interactive topics look good"""
    js_url = flask.url_for('static', filename='test1.js')
    css_url = flask.url_for('static', filename='test1.css')
    return flask.render_template("test1.html", jsurl=js_url, cssurl = css_url)

if __name__ == '__main__':
    get_newsgroups()
    app.run(debug=True, use_reloader=False)

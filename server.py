"""Runs a demo of the anchor words algorithm

This is really a proof of concept and is as ugly as sin.

Note that if you want to run this, you'll have to update the file paths in the
get_newsgroups function to be something useful to you. Those paths currently
point at a clone of github.com/jlund3/data, so you may want to clone my data
repo as well.
"""

import flask
import json
import numpy

import ankura
from ankura import util

app = flask.Flask(__name__, static_url_path='')


@util.memoize
@util.pickle_cache('newsgroups.pickle')
def get_newsgroups():
    """Retrieves the 20 newsgroups dataset"""
    news_glob = '/local/jlund3/data/newsgroups/*/*'
    engl_stop = '/local/jlund3/data/stopwords/english.txt'
    news_stop = '/local/jlund3/data/stopwords/newsgroups.txt'
    pipeline = [(ankura.read_glob, news_glob, ankura.tokenize.news),
                (ankura.filter_stopwords, engl_stop),
                (ankura.filter_stopwords, news_stop),
                (ankura.filter_rarewords, 100),
                (ankura.filter_commonwords, 1500)]
    print ' * Import pipeline running'
    dataset = ankura.run_pipeline(pipeline)
    print ' * Pipeline complete'
    return dataset

@util.memoize
@util.pickle_cache('anchors.pickle')
def default_anchors():
    """Retrieves default anchors for newsgroups using Gram-Schmidt"""
    return ankura.gramschmidt_anchors(get_newsgroups(), 20, 500)


@util.memoize
def get_topics(dataset, anchors):
    """Gets the topics for 20 newsgroups given a set of anchors"""
    return ankura.recover_topics(dataset, anchors)


@util.memoize
def reindex_anchors(dataset, anchors):
    """Converts any tokens in a set of anchors to the index of token"""
    conversion = lambda t: t if isinstance(t, int) else dataset.vocab.index(t)
    return util.tuplize(anchors, conversion)


@util.memoize
def tokenify_anchors(dataset, anchors):
    """Converts token indexes in a list of anchors to tokens"""
    return [[dataset.vocab[token] for token in anchor] for anchor in anchors]


@app.route('/topics')
def topic_request():
    """Performs a topic request using anchors from the query string"""
    dataset = get_newsgroups()
    raw_anchors = flask.request.args.get('anchors')

    if raw_anchors is None:
        anchors = default_anchors()
    else:
        anchors = util.tuplize(json.loads(raw_anchors))
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


if __name__ == '__main__':
    default_anchors()
    app.run(debug=True)

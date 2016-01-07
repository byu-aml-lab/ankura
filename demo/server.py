"""Runs a demo of an interactive anchor words algorithm

This is really a proof of concept and is ugly as sin.
"""
import flask
import json
import numpy

import ankura
from demo.example import get_newsgroups, default_anchors, get_topics

app = flask.Flask(__name__, static_url_path='')

@ankura.util.memoize
def reindex_anchors(dataset, anchors):
    """Converts any tokens in a set of anchors to the index of token"""
    conversion = lambda t: t if isinstance(t, int) else dataset.vocab.index(t)
    return ankura.util.tuplize(anchors, conversion)


@ankura.util.memoize
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
        anchors = ankura.util.tuplize(json.loads(raw_anchors))
    anchors = reindex_anchors(dataset, anchors)
    topics = get_topics(dataset, anchors)

    topic_summary = []
    for k in range(topics.shape[1]):
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


@app.route('/')
def root():
    """Serves up the single page app which demos interactive topics"""
    return flask.send_from_directory('.', 'index.html')


if __name__ == '__main__':
    default_anchors()
    app.run(debug=True)

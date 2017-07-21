#!/usr/bin/python3
"""Runs a user interface for interactive anchor words algorithm"""

import flask

import ankura
import ankura.util

app = flask.Flask(__name__, static_url_path='')


@ankura.util.memoize
def get_corpus():
    """Gets the 20 newsgroups corpus"""
    return ankura.corpus.newsgroups()


@ankura.util.memoize
def get_Q():
    """Gets the cooccurrence matrix for the corpus"""
    return ankura.anchor.build_cooccurrence(get_corpus())


@ankura.util.memoize
def get_gs_anchors(k=20):
    """Gets the default gram-schmidt anchors for the corpus"""
    corpus = get_corpus()
    Q = get_Q()
    indices = ankura.anchor.gram_schmidt(corpus, Q, k, return_indices=True)
    anchors = Q[indices, :]
    tokens = [[corpus.vocabulary[i]] for i in indices]
    return anchors, tokens


def get_topics(anchors):
    """Gets the topics from a set of anchors"""
    return ankura.anchor.recover_topics(get_Q(), anchors)


@app.route('/topics')
def topic_request():
    """Performs a topic request using anchors from the query string"""
    anchor_tokens = flask.request.args.get('anchors')
    if anchor_tokens is None:
        anchor_vectors, anchor_tokens = get_gs_anchors()
    else:
        corpus = get_corpus()
        Q = get_Q()
        anchor_vectors = ankura.anchor.tandem_anchors(anchor_tokens, Q, corpus)

    topics = get_topics(anchor_vectors)
    summary = ankura.topic.topic_summary(topics, get_corpus())

    # TODO Accuracy from free classifier

    return flask.jsonify(anchors=anchor_tokens,
                         topics=summary)


@app.route('/vocab')
def vocab_request():
    """Gets all valid vocabulary words in the corpus"""
    return flask.jsonify(vocab=get_corpus().vocabulary)


if __name__ == '__main__':
    # trigger memoization
    get_gs_anchors()
    # start dev server
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8000)

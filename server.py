"""Runs a demo of the anchor words algorithm"""

import flask
import json
import numpy

import ankura

app = flask.Flask(__name__)


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


@app.route('/topics')
def topic_request():
    """Performs a topic request using anchors from the query string"""
    dataset = get_newsgroups()
    print dataset
    anchors = json.loads(flask.request.args.get('anchors'))
    print anchors
    anchors = reindex_anchors(dataset, anchors)
    print anchors
    topics = get_topics(dataset, anchors)
    print topics

    topic_summary = []
    for k in xrange(topics.shape[1]):
        summary = []
        for word in numpy.argsort(topics[:, k])[-20:][::-1]:
            summary.append(dataset.vocab[word])
        topic_summary.append(summary)
    print topic_summary
    return flask.jsonify(topics=topic_summary)


@app.route('/dataset')
def data_request():
    """Gets the coocurrence matrix for the newsgroups dataset"""
    return flask.jsonify(data=get_newsgroups().Q.tolist())


if __name__ == '__main__':
    get_newsgroups()
    app.run(debug=True, use_reloader=False)

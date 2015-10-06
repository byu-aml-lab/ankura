"""Runs a demo of the anchor words algorithm"""

import flask
import json

import ankura

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
    return ankura.run_pipeline(pipeline)


@memoize
def get_topics(anchors):
    """Gets the topics for 20 newsgroups given a set of anchors"""
    return ankura.recover_topics(get_newsgroups(), anchors)


app = flask.Flask(__name__)

@app.route('/topics')
def topic_request():
    """Performs a topic request using anchors from the query string"""
    return get_topics(json.loads(flask.request.args.get('anchors')))


@app.route('/dataset')
def data_request():
    """Gets the coocurrence matrix for the newsgroups dataset"""
    return get_newsgroups().Q


if __name__ == '__main__':
    get_newsgroups()
    app.run()

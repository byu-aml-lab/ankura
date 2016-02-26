"""Runs a user interface for the interactive anchor words algorithm"""

import flask
import json
import numpy
import numbers
import tempfile
import os

import ankura

app = flask.Flask(__name__, static_url_path='')


def convert_anchor(dataset, anchor):
    """Converts an anchor it its integer index"""
    if isinstance(anchor, numbers.Integral):
        return anchor
    else:
        return dataset.vocab.index(anchor)

@ankura.util.memoize
@ankura.util.pickle_cache('fcc.pickle')
def get_newsgroups():
    """Retrieves the 20 newsgroups dataset"""
    filenames = '/local/cojoco/git/fcc/documents/*.txt'
#    news_glob = '/local/cojoco/git/jeffData/newsgroups/*/*'
    engl_stop = '/local/cojoco/git/jeffData/stopwords/english.txt'
    news_stop = '/local/cojoco/git/jeffData/stopwords/newsgroups.txt'
    name_stop = '/local/cojoco/git/fcc/stopwords/names.txt'
    curse_stop = '/local/cojoco/git/jeffData/stopwords/profanity.txt'
    pipeline = [(ankura.read_glob, filenames, ankura.tokenize.news),
                (ankura.filter_stopwords, engl_stop),
                (ankura.filter_stopwords, news_stop),
                (ankura.combine_words, name_stop, '<name>', ankura.tokenize.simple),
                (ankura.combine_words, curse_stop, '<profanity>', ankura.tokenize.simple),
                (ankura.filter_rarewords, 200),
                (ankura.filter_commonwords, 150000)]
    dataset = ankura.run_pipeline(pipeline)
    return dataset

@ankura.util.memoize
@ankura.util.pickle_cache('fcc-anchors-default.pickle')
def default_anchors():
    """Retrieves default anchors for newsgroups using Gram-Schmidt"""
    return ankura.gramschmidt_anchors(get_newsgroups(), 20, 500)

@ankura.util.memoize
def get_topics(dataset, anchors):
    """Gets the topics for 20 newsgroups given a set of anchors"""
    return ankura.recover_topics(dataset, anchors)

@ankura.util.memoize
def reindex_anchors(dataset, anchors):
    """Converts any tokens in a set of anchors to the index of token"""
    conversion = lambda t: convert_anchor(dataset, t)
    return ankura.util.tuplize(anchors, conversion)


@ankura.util.memoize
def tokenify_anchors(dataset, anchors):
    """Converts token indexes in a list of anchors to tokens"""
    return [[dataset.vocab[token] for token in anchor] for anchor in anchors]

@app.route('/base-anchors')
def get_base_anchors():
    """Gets the base set of anchors to send to the client"""
    base_anchors = default_anchors()
    return flask.jsonify(anchors=tokenify_anchors(get_newsgroups(), base_anchors))

@app.route('/topics')
def topic_request():
    """Performs a topic request using anchors from the query string"""
    dataset = get_newsgroups()
    raw_anchors = flask.request.args.get('anchors')

    if raw_anchors is None:
        anchors = default_anchors()
    else:
        print(json.loads(raw_anchors))
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
def serve_itm():
    """Serves the Interactive Topic Modeling UI"""
    return flask.send_from_directory('static', 'index.html')

@app.route('/style/stylesheet.css')
def serve_itm_css():
    """Serves the CSS for the ITM UI"""
    return flask.send_from_directory('static/style', 'stylesheet.css')

@app.route('/scripts/script.js')
def serve_itm_js():
    """Serves the Javascript for the ITM UI"""
    return flask.send_from_directory('static/scripts', 'script.js')

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

@app.route('/test3')
def serve_test3():
    """Serves the test for the anchor words algorithm done in Javascript"""
    return flask.send_from_directory('static', 'test3.html')

@app.route('/style/test3.css')
def serve_test3_css():
    """Serves the CSS for the test anchor words in Javascript"""
    return flask.send_from_directory('static/style', 'test3.css')

@app.route('/scripts/test3.js')
def serve_test3_js():
    """Serves the Javascript for the test anchor words in Javascript"""
    return flask.send_from_directory('static/scripts', 'test3.js')

@app.route('/scripts/linear.js')
def serve_linear_js():
    """Serves the Javascript methods for the anchor word algorithm"""
    return flask.send_from_directory('static/scripts', 'linear.js')

@app.route('/vocab')
def get_vocab():
    """Returns all valid vocabulary words in the dataset"""
    dataset = get_newsgroups()
    return flask.jsonify(vocab=dataset.vocab)

@app.route('/vocabsize')
def get_vocab_size():
    """Returns the size of the vocab being used"""
    dataset = get_newsgroups()
    return "Vocabulary Size: " + str(dataset.vocab_size)

@app.route('/cooccurrences')
def get_cooccurrences():
    """Returns the cooccurrences matrix from the dataset"""
    dataset = get_newsgroups()
    return flask.jsonify(cooccurrences=dataset.Q.tolist())

@app.route('/images/spinner.gif')
def get_spinner():
    """Serves the spinning wheel gif"""
    return flask.send_from_directory('static/images', 'spinner.gif')

if __name__ == '__main__':
    default_anchors()
    app.run(debug=True,
            host='0.0.0.0')

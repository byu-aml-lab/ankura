"""A collection of labelers for use ankura import pipelines"""

import os

from . import tokenize

# Note: Each labeler takes in title and data and should return a map of
# metadata key/value pairs.


def aggregate(*labelers):
    """Aggregates multiple labelers into a single labeler"""
    def aggregated(title, data):
        """Calls multiple labelers and aggregates thier results"""
        metadata = {}
        for labeler in labelers:
            metadata.update(labeler(title, data))
        return metadata
    return aggregated


def text(_, data, formatter=None):
    """Labels the document with its original untokenized text"""
    if formatter:
        data = formatter(data)
    return {'text': data.strip()}


def title_dirname(title, _):
    """Treating the title as a filepath, labels documents with title dirname"""
    return {'dirname': os.path.dirname(title)}


def news_formatter(data):
    """Formats data by skipping a file header"""
    return tokenize.news(data, tokenize.noop)


def html_formatter(data):
    """Formats data by skipping a file header"""
    return tokenize.html(data, tokenize.noop)

"""A collection of labelers for use ankura import pipelines"""

import os

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


def text(_, data):
    """Labels the document with its original untokenized text"""
    return {'text': data.strip()}


def title_dirname(title, _):
    """Treating the title as a filepath, labels documents with title dirname"""
    return {'dirname': os.path.dirname(title)}

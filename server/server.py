#!/usr/bin/python3
"""Runs a user interface for interactive anchor words algorithm"""

import functools
import gzip
import json

import ankura


def amazon_json_extractor():
    """Reads the amazon large json file"""
    @functools.wraps(amazon_json_extractor)
    def _extractor(docfile):
        for line in docfile:
            review = json.loads(line.decode())
            name = review['asin'] + '-' + review['reviewerID']
            data = review['summary'] + ' ' + review['reviewText']
            yield ankura.pipeline.Text(name, data)
    return _extractor


def amazon_labeler(pipeline):
    """Reads labels for amazon"""
    metadata = {}
    for docfile in pipeline.inputer():
        docfile = gzip.GzipFile(fileobj=docfile)
        for line in docfile:
            review = json.loads(line.decode())
            name = review['asin'] + '-' + review['reviewerID']
            rating = float(review['overall'])
            metadata[name] = {'rating': rating, 'asin': review['asin']}
    @functools.wraps(amazon_labeler)
    def _labeler(name):
        return metadata[name]
    return _labeler


def amazon():
    """Gets the amazon large dataset"""
    datapath = '/aml/data/amazon_large/item_dedup.json.gz'
    stoppath = '/aml/data/stopwords/english.txt'
    picklepath = '/aml/scratch/jlund3/amazon.pickle'
    pipeline = ankura.pipeline.Pipeline(
        ankura.pipeline.file_inputer(datapath),
        ankura.pipeline.gzip_extractor(amazon_json_extractor()),
        ankura.pipeline.stopword_tokenizer(
            ankura.pipeline.default_tokenizer(),
            open(stoppath),
        ),
        ankura.pipeline.noop_labeler(),
        ankura.pipeline.length_filterer(),
    )
    pipeline.tokenizer = ankura.pipeline.frequency_tokenizer(pipeline, 50)
    pipeline.labeler = amazon_labeler(pipeline)
    return pipeline.run(picklepath)


if __name__ == '__main__':
    amazon()

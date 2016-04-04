"""A collection of tokenizers for use with ankura import pipelines"""

import re

import bs4

# Note: Each tokenizer takes in a string, and returns a list of tokens


def split(data):
    """A tokenizer which does nothing but splitting"""
    return data.split()


def simple(data, splitter=split):
    """A basic tokenizer which splits and does basic filtering.

    The included filters and transformations include:
    * lower case each token
    * filter out non-alphabetic characters
    """
    tokens = splitter(data)
    tokens = [token.lower() for token in tokens]
    tokens = [re.sub(r'[^a-z]', '', token) for token in tokens]
    tokens = [token for token in tokens if token]
    return tokens


def news(data, tokenizer=simple):
    """Tokenizes after skipping a file header

    Using the format from the well-known 20 newsgroups dataset, we consider the
    header to be everything before the first empty line in the file. The
    remaining contents of the file are then tokenized.
    """
    match = re.search(r'\n\s*\n', data, re.MULTILINE)
    if match:
        data = data[match.end():]
    return tokenizer(data)


def html(data, tokenizer=simple):
    """Tokenizes by extracting text from an HTML file"""
    return tokenizer(bs4.BeautifulSoup(data, 'html.parser').get_text())


def noop(data):
    """A noop tokenizer"""
    return data

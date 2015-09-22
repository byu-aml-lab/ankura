"""A collection of tokenizers for use with ankura import pipelines"""

import re
import io


def split(doc_file):
    """A tokenizer which does nothing but splitting"""
    return doc_file.read().split()


def simple(doc_file, splitter=split):
    """A basic tokenizer which splits and does basic filtering.

    The included filters and transofmrations include:
    * lower case each token
    * filter out non-alphabetic characters
    * filter out words with fewer than 3 characters
    * filter out words with more than 20 characters
    """
    tokens = splitter(doc_file)
    tokens = [token.lower() for token in tokens]
    tokens = [re.sub(r'[^a-z]', '', token) for token in tokens]
    tokens = [token for token in tokens if 3 < len(token) < 20]
    return tokens


def news(doc_file, tokenizer=simple):
    """Tokenizes after skipping a file header

    Using the format from the well-known 20 newsgroups dataset, we consider the
    header to be everything before the first empty line in the file. The
    remaining contents of the file are then tokenized.
    """
    # skip header by finding first empty line
    line = doc_file.readline()
    while line.strip():
        line = doc_file.readline()
    # use tokenizer on what remains in file
    return tokenizer(doc_file)


def html(doc_file, tokenizer=simple):
    """Tokenizes by extracting text from an HTML file"""
    # parse the text in the html
    text = doc_file.read().strip()
    text = re.sub(r'(?is)<(script|style).*?>.*?(</\1>)', '', text)
    text = re.sub(r'(?s)<!--(.*?)-->[\n]?', '', text)
    text = re.sub(r'(?s)<.*?>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'  ', ' ', text)
    text = re.sub(r'  ', ' ', text)
    text = text.strip()
    # tokenize the parsed text
    return tokenizer(io.StringIO(text))

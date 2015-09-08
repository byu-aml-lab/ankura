"""A collection of tokenizers for use with ankura import pipelines"""

import re
import io

def simple(doc_file):
    """A basic tokenizer which splits and filters text without preprocessing"""
    tokens = doc_file.read().split()
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    tokens = [token for token in tokens if 2 < len(token) < 10]
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



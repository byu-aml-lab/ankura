"""A collection of segmenters for use with ankura import pipelines"""

import re
import os

import bs4

# Note: Each segmenter takes in a file object and should return an iterable of
# tuples containing title and text as strings.

def simple(docfile):
    """Considers the file to be a single document"""
    return [(docfile.name, docfile.read())]


def lines(docfile):
    """Segments the text so that each line becomes to a single document

    The segmented data is returned as a generator so that the entire contents
    of the file are not nessesarily stored in memory depending on how the
    segmenter is used downstream. The first sequence of letters unbroken by
    whitespace will be considered the title of the document, and the remainder
    of each line will become the text of each document. The given data must be
    an iterable over the lines of the data so an opened file (whose iterator
    yields the lines of the file) will work, as will any iterator over str.
    """
    return (line.split(None, 1) for line in docfile)


SECTION_RE = re.compile(r'^((([Ss]ection)|(SECTION))\s+)?\d+(\.\d+){1,2}')

def _sort_sections(section_names):
    return sorted(section_names, key=lambda n: int(re.sub(r'[^\d]', '', n)))


def section(docfile):
    """Segments a legal contract into sections

    Assumes that all sections start with something like 'Section 2.03' or
    '3.2'. If this assumption is false, then the segmenter will incorrectly
    break up the contract.
    """
    soup = bs4.BeautifulSoup(docfile, 'html.parser')

    sections = {}

    section = []
    for p in soup.find_all('p'):
        text = p.get_text().strip()

        if SECTION_RE.match(text):
            section_text = '\n'.join(section)
            section_num = SECTION_RE.match(section_text)
            if section_num:
                sections[section_num.group()] = section_text
            section = []

        if text:
            section.append(text)

    for section_num in _sort_sections(sections):
        title = os.path.join(docfile.name, section_num)
        yield title, sections[section_num]

    # TODO(jlund) Misses non-labeled section in Article I

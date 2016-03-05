"""A collection of segmenters for use with ankura import pipelines"""

def lines(lines):
    """Segements the text so that each line becomes to a single document

    The segmented data is returned as a generator so that the entire contents
    of the file are not nessesarily stored in memory depending on how the
    segmenter is used downstream. The first sequence of letters unbroken by
    whitespace will be considered the title of the document, and the remainder
    of each line will become the text of each document. The given data must be
    an iterable over the lines of the data so an opened file (whose iterator
    yields the lines of the file) will work, as will any iterator over str.
    """
    return (line.split(None, 1) for line in lines)

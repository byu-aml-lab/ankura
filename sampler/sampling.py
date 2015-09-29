from __future__ import division
import ctypes
import numpy as np
import sklearn.linear_model as linear_model
import sys

def init_topic_assignments(doc_ids, dataset, rnd, numtopics):
    result = (ctypes.POINTER(ctypes.c_int) * len(doc_ids))()
    for i in range(len(doc_ids)):
        name = doc_ids[i]
        tmp = (ctypes.c_int * len(dataset.doc_tokens(name)))()
        for j in range(len(tmp)):
            tmp[j] = rnd.randint(0, numtopics-1)
        result[i] = tmp
    return result

def count_topic_assignments(doc_ids, numtopics, vocabsize, dataset,
        topicassignments, wordindex):
    doctopic = (ctypes.POINTER(ctypes.c_int) * len(doc_ids))()
    topicword = (ctypes.POINTER(ctypes.c_int) * numtopics)()
    for i in range(numtopics):
        topicword[i] = (ctypes.c_int * vocabsize)()
    for i in range(len(topicassignments)):
        docWords = dataset.doc_tokens(doc_ids[i])
        doctopic[i] = (ctypes.c_int * numtopics)()
        for j in range(len(docWords)):
            topic = topicassignments[i][j]
            doctopic[i][topic] += 1
            topicword[topic][wordindex.get_index(docWords[j])] += 1
    return doctopic, topicword


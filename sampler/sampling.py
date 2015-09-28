from __future__ import division
import ctypes
import numpy as np
import sklearn.linear_model as linear_model
import sys

def init_topic_assignments(names, corpus, rnd, numtopics):
    result = (ctypes.POINTER(ctypes.c_int) * len(names))()
    for i in range(len(names)):
        name = names[i]
        tmp = (ctypes.c_int * len(corpus[name]))()
        for j in range(len(tmp)):
            tmp[j] = rnd.randint(0, numtopics-1)
        result[i] = tmp
    '''
    with open('inittopics.txt', 'w') as ofh:
        for r in result:
            ofh.write(name)
            ofh.write('\t')
            ofh.write(str(result[-1]))
            ofh.write('\n')
    sys.exit()
    '''
    return result

def count_topic_assignments(names, numtopics, vocabsize, corpus,
        topicassignments, wordindex):
    # print('==== count_topic_assignments ====')
    doctopic = (ctypes.POINTER(ctypes.c_int) * len(names))()
    topicword = (ctypes.POINTER(ctypes.c_int) * numtopics)()
    for i in range(numtopics):
        topicword[i] = (ctypes.c_int * vocabsize)()
    for i in range(len(topicassignments)):
        docWords = corpus[names[i]]
        doctopic[i] = (ctypes.c_int * numtopics)()
        for j in range(len(docWords)):
            topic = topicassignments[i][j]
            doctopic[i][topic] += 1
            topicword[topic][wordindex.get_index(docWords[j])] += 1
    '''
    for i in range(numtopics):
        for j in range(vocabsize):
            print(i, j, topicword[i][j])
    print('==== end count_topic_assignments ====')
    '''
    return doctopic, topicword


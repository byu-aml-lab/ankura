from __future__ import division
import ctypes

class WordIndex:
    def __init__(self):
        self.wordtoindex = {}
        self.indextoword = []
    def index(self, word):
        if word not in self.wordtoindex:
            self.wordtoindex[word] = len(self.indextoword)
            self.indextoword.append(word)
    def get_index(self, word):
        if word not in self.wordtoindex:
            return -1
        return self.wordtoindex[word]
    def get_word(self, index):
        return self.indextoword[index]
    def size(self):
        return len(self.indextoword)
    def vectorize(self, text):
        result = []
        for word in text:
            self.index(word)
            result.append(self.get_index(word))
        return result
    def vectorize_without_adding(self, text):
        result = []
        for word in text:
            index = self.get_index(word)
            if index > -1:
                result.append(index)
        return result

def vectorize_training(names, dataset):
    trainingvectors = (ctypes.POINTER(ctypes.c_int) * len(names))()
    wordindex = WordIndex()
    for i in range(len(names)):
        name = names[i]
        curvector = wordindex.vectorize(dataset.doc_tokens(name))
        convertedvector = (ctypes.c_int * len(curvector))()
        for j in range(len(curvector)):
            convertedvector[j] = ctypes.c_int(curvector[j])
        trainingvectors[i] = convertedvector
    return trainingvectors, wordindex 

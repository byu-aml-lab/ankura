from __future__ import division
import copy
import ctypes
import numpy as np
import os
import random
import sys

import sampling as sampling
import wordindex as wordindex
import ctypesutils as ctypesutils

script_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(script_dir, 'csampling.so')
sampling_dll = ctypes.CDLL(so_path)

class CorpusData(ctypes.Structure):
    _fields_ = [
        ('numDocs', ctypes.c_int),
        ('docSizes', ctypes.POINTER(ctypes.c_int)),
        ('docWords', ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
        ('numResponses', ctypes.c_int),
        ('responseValues', ctypes.POINTER(ctypes.c_double))
    ]

class SamplerState(ctypes.Structure):
    _fields_ = [
        ('numTopics', ctypes.c_int),
        ('vocabSize', ctypes.c_int),
        ('alphas', ctypes.POINTER(ctypes.c_double)),
        ('hyperbeta', ctypes.c_double),
        ('eta', ctypes.POINTER(ctypes.c_double)),
        ('var', ctypes.c_double),
        ('corpusData', ctypes.POINTER(CorpusData)),
        ('topicAssignments', ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
        ('docTopicCounts', ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
        ('topicWordCounts', ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
        ('topicWordSum', ctypes.POINTER(ctypes.c_int))
    ]

sampling_sLDA = sampling_dll.sample_sLDA
sampling_sLDA.argtypes = (ctypes.POINTER(SamplerState), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int))
sampling_sLDA.restype = ctypes.POINTER(ctypes.POINTER(SamplerState))
sampling_setSeed = sampling_dll.setSeed
sampling_setSeed.argtypes = (ctypes.c_ulonglong,)
getExpectedTopicCounts = sampling_dll.getExpectedTopicCounts
getExpectedTopicCounts.argtypes = (ctypes.POINTER(SamplerState), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int))
getExpectedTopicCounts.restype = ctypes.POINTER(ctypes.c_double)
freeSavedStates = sampling_dll.freeSavedStates
freeSavedStates.argtypes = (ctypes.POINTER(ctypes.POINTER(SamplerState)),
        ctypes.c_int)
cPredict = sampling_dll.predict
cPredict.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.POINTER(SamplerState)),
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int))
cPredict.restype = ctypes.POINTER(ctypes.c_double)
freeDoubleArray = sampling_dll.freeDoubleArray
freeDoubleArray.argtypes = (ctypes.POINTER(ctypes.c_double),)

def set_seed(seed):
    sampling_setSeed(ctypes.c_ulonglong(seed))

def build_model(modelparams, rnd):
    modeltype = modelparams['modeltype']
    if modeltype == 'MockModel':
        return MockModel(rnd)
    elif modeltype == 'SamplingSLDA':
        return SamplingSLDA(rnd,
                int(modelparams['model_numtopics']),
                float(modelparams['model_initalpha']),
                float(modelparams['model_inithyperbeta']),
                float(modelparams['model_initvar']),
                int(modelparams['model_numtrainchains']),
                int(modelparams['model_numsamplespertrainchain']),
                int(modelparams['model_trainburn']),
                int(modelparams['model_trainlag']),
                int(modelparams['model_numsamplesperpredictchain']),
                int(modelparams['model_predictburn']),
                int(modelparams['model_predictlag']))
    raise NotImplementedError('Unknown model type: \
            {}'.format('modeltype'))

class AbstractModel:
    def train(self, trainingset, knownlabels):
        raise NotImplementedError()
    def predict(self, doc):
        raise NotImplementedError()

class MockModel(AbstractModel):
    '''MockModel requires no parameters to be given'''
    def __init__(self, rnd):
        self.rnd = rnd
    def train(self, dataset, knownlabels):
        pass
    def predict(self, doc):
        return self.rnd.random()

class SamplingSLDA(AbstractModel):
    '''SamplingSLDA requires the following parameters:
        * numtopics
            the number of topics to look for
        * initalpha
            the initial value for the hyperparameter over the per-document topic
            distribution
        * inithyperbeta
            the initial value for the hyperparameter over the per topic
            word-type distribution
        * initvar
            the initial value for the variance parameter of the prediction
            variables
        * numtrainchains
            the number of sampling chains to use during training
        * numsamplespertrainchain
            the number of samples to keep per training chain
        * trainburn
            the number of sampling iterations to burn during training
        * trainlag
            the number of sampling iterations between samples kept during
            training
        * numsamplesperpredictchain
            the number of samples to keep per prediction chain; each sample
            saved during training will spawn a prediction chain
        * predictburn
            the number of sampling iterations to burn during prediction
        * predictlag
            the number of sampling iterations between samples kept during
            prediction
    '''
    def __init__(self, rnd, numtopics, initalpha, inithyperbeta, initvar, numtrainchains,
            numsamplespertrainchain, trainburn, trainlag,
            numsamplesperpredictchain, predictburn, predictlag):
        self.rnd = rnd
        self.numtopics = numtopics
        self.alphas = (ctypes.c_double * numtopics)()
        for i in range(numtopics):
            self.alphas[i] = initalpha
        self.hyperbeta = ctypes.c_double(inithyperbeta)
        self.var = ctypes.c_double(initvar)
        self.numtrainchains = numtrainchains
        self.numsamplespertrainchain = numsamplespertrainchain
        self.trainburn = trainburn
        self.trainlag = trainlag
        self.numsamplesperpredictchain = numsamplesperpredictchain
        self.predictburn = predictburn
        self.predictlag = predictlag
        self.saved_statesc = [0] * self.numtrainchains

        predictschedule = [self.predictlag] * self.numsamplesperpredictchain
        predictschedule[0] = self.predictburn
        self.predictschedarr = ctypesutils.convertFromIntList(predictschedule)

        # other instance variables initialized in train:
        #   self.trainingdoc_ids
        #   self.trainvectors
        #   self.wordindex
        #   self.prevlabeledcount

    def train(self, dataset, trainingdoc_ids, knownresp, continue_training=False):
        # trainingdoc_ids must be a list
        # knownresp must be a list such that its values correspond with trainingdoc_ids
        self.trainingdoc_ids = copy.deepcopy(trainingdoc_ids)
        responseValues = ctypesutils.convertFromDoubleList(knownresp)
        sizesList = []
        for doc_id in self.trainingdoc_ids:
            sizesList.append(len(dataset.doc_tokens(doc_id)))
        docSizes = ctypesutils.convertFromIntList(sizesList)
        trainvectors, self.wordindex = \
                wordindex.vectorize_training(self.trainingdoc_ids, dataset)
        vocabSize = ctypes.c_int(self.wordindex.size())
        numVocabList = [self.wordindex.size()] * self.numtopics
        corpusData = CorpusData(
                len(self.trainingdoc_ids), docSizes, trainvectors,
                len(knownresp), ctypesutils.convertFromDoubleList(knownresp))
        loop_schedule = [self.trainlag] * self.numsamplespertrainchain

        for curchain in range(self.numtrainchains):
            topicassignments = sampling.init_topic_assignments(
                    self.trainingdoc_ids, dataset, self.rnd, self.numtopics)
            eta = (ctypes.c_double * self.numtopics)()
            if continue_training:
                # fill back previous state of sampler
                final_state = self.numsamplespertrainchain - 1
                prev_assignments = self.saved_statesc[curchain][final_state].contents.topicAssignments
                for i in range(self.prevlabeledcount):
                    # assuming that the first self.prevlabeledcount elements of
                    # self.trainingdoc_ids are the same labeled documents trained
                    # on in the last training iteration
                    for j in range(len(dataset.doc_tokens(self.trainingdoc_ids[i]))):
                        topicassignments[i][j] = prev_assignments[i][j]
                for i in range(self.numtopics):
                    eta[i] = self.saved_statesc[curchain][final_state].contents.eta[i]
                self.var = self.saved_statesc[curchain][final_state].contents.var
                freeSavedStates(self.saved_statesc[curchain],
                        ctypes.c_int(self.numsamplespertrainchain))
            else:
                # need to set first loop to go through burn
                loop_schedule[0] = self.trainburn
                labelsmean = np.mean(knownresp)
                for i in range(self.numtopics):
                    eta[i] = ((float(i*2) - 1.0) / (float(self.numtopics) - 1.0)) \
                            + labelsmean
            doctopiccounts, topicwordcounts = \
                    sampling.count_topic_assignments(self.trainingdoc_ids, self.numtopics,
                            self.wordindex.size(), dataset, topicassignments,
                            self.wordindex)

            prepresums = np.array(ctypesutils.convertToListOfLists(topicwordcounts,
                        numVocabList))
            presums = np.sum(prepresums, axis=1)
            topicwordsum = ctypesutils.convertFromIntList(presums)
            samplerState = SamplerState(self.numtopics, self.wordindex.size(),
                    self.alphas, self.hyperbeta, eta, self.var,
                    ctypes.pointer(corpusData),
                    topicassignments, doctopiccounts, topicwordcounts,
                    topicwordsum)

            self.saved_statesc[curchain] = sampling_sLDA(samplerState, ctypes.c_int(self.numsamplespertrainchain),
                    ctypesutils.convertFromIntList(loop_schedule))
        self.prevlabeledcount = len(self.trainingdoc_ids)

    def predict(self, doc):
        resultsList = []
        docws = self.wordindex.vectorize_without_adding(doc)
        if len(docws) <= 0:
            return self.rnd.random()
        docsize = len(docws)
        docwsarr = ctypesutils.convertFromIntList(docws)
        for curchain in range(self.numtrainchains):
            cResults = cPredict(ctypes.c_int(self.numsamplespertrainchain),
                    self.saved_statesc[curchain], ctypes.c_int(docsize),
                    docwsarr,
                    self.numsamplesperpredictchain,
                    self.predictschedarr)
            # get a query by committee result
            resultsList.append(np.mean(ctypesutils.convertToList(cResults,
                    self.numsamplespertrainchain *
                    self.numsamplesperpredictchain)))
            freeDoubleArray(cResults)
        return np.mean(resultsList)

    def cleanup(self):
        count = ctypes.c_int(self.numsamplespertrainchain)
        for i in range(self.numtrainchains):
            freeSavedStates(self.saved_statesc[i], count)
    
    def get_expected_topic_counts(self, words, state_num):
        expectedTopicCounts = getExpectedTopicCounts(self.saved_statesc[state_num],
                ctypes.c_int(len(words)), ctypesutils.convertFromIntList(words),
                self.numsamplesperpredictchain, self.predictschedarr)
        result = np.array(ctypesutils.convertToList(expectedTopicCounts,
                self.numtopics))
        freeDoubleArray(expectedTopicCounts)
        return result

    def get_topic_distribution(self, topic, state_num):
        result = np.array(ctypes.convertToList(self.saved_statesc[state_num].topicWordCounts[topic]))
        return result / np.sum(result)

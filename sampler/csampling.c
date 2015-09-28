#include "lapacke.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// from https://en.wikipedia.org/wiki/Xorshift
#include <stdint.h>

uint64_t rand_x = 8936517; /* The state must be seeded with a nonzero value. */

uint64_t xorshift64star(void) {
    rand_x ^= rand_x >> 12; // a
    rand_x ^= rand_x << 25; // b
    rand_x ^= rand_x >> 27; // c
    return rand_x * UINT64_C(2685821657736338717);
}
// end wikipedia

const double RAND_COEFF = 1.0 / UINT64_MAX;

double nextDouble(void) {
    return xorshift64star() * RAND_COEFF;
}

unsigned int nextUnsignedInt(unsigned int max) {
    return ((unsigned int)xorshift64star()) % max;
}

void setSeed(uint64_t seed) {
    rand_x = seed;
}

typedef struct corpusData_st {
    // numDocs != numResponses in the case of semi-supervised
    int numDocs;
    int * docSizes;
    int ** docWords;
    int numResponses;
    double * responseValues;
} CorpusData;

void freeCorpusData(CorpusData * cd) {
    const int numDocs = cd->numDocs;
    for (int i = 0; i < numDocs; i++) {
        free(cd->docWords[i]);
    }
    free(cd->docWords);
    free(cd->docSizes);
    free(cd->responseValues);
    free(cd);
}

CorpusData * copyCorpusData(CorpusData * cd) {
    CorpusData * result = malloc(sizeof(CorpusData));
    result->numDocs = cd->numDocs;
    result->docSizes = malloc(result->numDocs * sizeof(int));
    memcpy(result->docSizes, cd->docSizes, result->numDocs * sizeof(int));
    result->docWords = malloc(result->numDocs * sizeof(int *));
    for (int i = 0; i < result->numDocs; i++) {
        result->docWords[i] = malloc(result->docSizes[i] * sizeof(int));
        memcpy(result->docWords[i], cd->docWords[i], result->docSizes[i] * sizeof(int));
    }
    result->numResponses = cd->numResponses;
    result->responseValues = malloc(result->numResponses * sizeof(double));
    memcpy(result->responseValues, cd->responseValues, result->numResponses * sizeof(double));
    return result;
}

typedef struct samplerState_st {
    int numTopics;
    int vocabSize;
    double * alphas;
    double hyperbeta;
    double * eta;
    double var;
    CorpusData * corpusData;
    int ** topicAssignments;
    int ** docTopicCounts;
    int ** topicWordCounts;
    int * topicWordSum;
} SamplerState;

SamplerState * copySamplerState(SamplerState * model) {
    SamplerState * result = malloc(sizeof(SamplerState));
    result->numTopics = model->numTopics;
    const int numTopics = result->numTopics;
    result->vocabSize = model->vocabSize;
    const int vocabSize = result->vocabSize;
    const int alphasSize = numTopics * sizeof(double);
    result->alphas = malloc(alphasSize);
    memcpy(result->alphas, model->alphas, alphasSize);
    result->hyperbeta = model->hyperbeta;
    result->eta = malloc(alphasSize);
    memcpy(result->eta, model->eta, alphasSize);
    result->var = model->var;
    result->corpusData = copyCorpusData(model->corpusData);
    const int numDocs = result->corpusData->numDocs;
    result->topicAssignments = malloc(numDocs * sizeof(int *));
    for (int i = 0; i < numDocs; i++) {
        int assignments_iSize = result->corpusData->docSizes[i] * sizeof(int);
        result->topicAssignments[i] = malloc(assignments_iSize);
        memcpy(result->topicAssignments[i], model->topicAssignments[i], assignments_iSize);
    }
    const int topicCountSize = numTopics * sizeof(int);
    result->docTopicCounts = malloc(numDocs * sizeof(int *));
    for (int i = 0; i < numDocs; i++) {
        result->docTopicCounts[i] = malloc(topicCountSize);
        memcpy(result->docTopicCounts[i], model->docTopicCounts[i], topicCountSize);
    }
    const int wordCountSize = vocabSize * sizeof(int);
    result->topicWordCounts = malloc(numTopics * sizeof(int *));
    for (int i = 0; i < numTopics; i++) {
        result->topicWordCounts[i] = malloc(wordCountSize);
        memcpy(result->topicWordCounts[i], model->topicWordCounts[i], wordCountSize);
    }
    result->topicWordSum = malloc(numTopics * sizeof(int));
    memcpy(result->topicWordSum, model->topicWordSum, numTopics * sizeof(int));
    return result;
}

void freeSamplerState(SamplerState * samplerState) {
    free(samplerState->alphas);
    free(samplerState->eta);
    for (int i = 0; i < samplerState->corpusData->numDocs; i++) {
        free(samplerState->topicAssignments[i]);
        free(samplerState->docTopicCounts[i]);
    }
    free(samplerState->topicAssignments);
    free(samplerState->docTopicCounts);
    freeCorpusData(samplerState->corpusData);
    for (int i = 0; i < samplerState->numTopics; i++) {
        free(samplerState->topicWordCounts[i]);
    }
    free(samplerState->topicWordCounts);
    free(samplerState->topicWordSum);
    free(samplerState);
}

void freeSavedStates(SamplerState ** savedStates, int stateCount) {
    for (int i = 0; i < stateCount; i++) {
        freeSamplerState(savedStates[i]);
    }
    free(savedStates);
}

int sample_categorical(double * weights, int numTopics) {
    double weightsSum = 0.0;
    for (int i = 0; i < numTopics; i++) {
        weightsSum += weights[i];
    }
    double u = nextDouble() * weightsSum;
    double sampleSum = weights[0];
    int sample = 0;
    while (sample < numTopics && sampleSum < u) {
        sample += 1;
        sampleSum += weights[sample];
    }
    return sample;
}

// making docSize of type double so that division later won't truncate
double * sLDA_probability_calculator(int v, double vbeta,
        double docSize, int numTopics, int * docTopicCounts_d, int ** topicWordCounts,
        int * topicWordSum, double * alphas, double hyperbeta, double * eta, double var, double responseValue) {
    double topic_probs_topic[numTopics];
    for (int i = 0; i < numTopics; i++) {
        double njv_sum_i = topicWordSum[i] + vbeta;
        topic_probs_topic[i] = ((topicWordCounts[i][v]+hyperbeta) / njv_sum_i) * (docTopicCounts_d[i] + alphas[i]);
    }
    double sampleMean = 0.0;
    for (int i = 0; i < numTopics; i++) {
        sampleMean += eta[i] * ((double)docTopicCounts_d[i]) / docSize;
    }
    double ys_dMinusMean = responseValue - sampleMean;
    double minLogMetadataContribution = -INFINITY;
    double topic_probs_metadata[numTopics];
    for (int i = 0; i < numTopics; i++) {
        double eta_i_over_Nd = eta[i] / docSize;
        topic_probs_metadata[i] = eta_i_over_Nd * (ys_dMinusMean - (eta_i_over_Nd / 2.0)) / var;
        if (topic_probs_metadata[i] > minLogMetadataContribution)
            minLogMetadataContribution = topic_probs_metadata[i];
    }
    double * result = malloc(numTopics * sizeof(double));
    for (int i = 0; i < numTopics; i++) {
        result[i] = topic_probs_topic[i] * exp(topic_probs_metadata[i] - minLogMetadataContribution);
    }
    return result;
}

double * LDA_probability_calculator(int v, double vbeta,
        double docSize, int numTopics, int * docTopicCounts_d, int ** topicWordCounts,
        int * topicWordSum, double * alphas, double hyperbeta, double * eta, double var, double responseValue) {
    double * result = malloc(numTopics * sizeof(double));
    double njv_sum[numTopics];
    for (int i = 0; i < numTopics; i++) {
        njv_sum[i] = vbeta + topicWordSum[i];
    }
    for (int i = 0; i < numTopics; i++) {
        result[i] = ((topicWordCounts[i][v] + hyperbeta) / njv_sum[i]) * (docTopicCounts_d[i] + alphas[i]);
    }
    return result;
}

void gibbs_inner_loop(int numTopics, double vbeta, int docSize,
        int * topicAssignments_d, int * docWords_d, int * docTopicCounts_d,
        int ** topicWordCounts, int * topicWordSum, double * alphas, double hyperbeta,
        double * eta, double var, double response_d,
        double * (* probability_calculator)(int, double, double, int, int *, int **, int *, double *, double, double *, double, double)) {
    // randomize sampling order
    int sampleOrder[docSize];
    for (int i = 0; i < docSize; i++) {
        sampleOrder[i] = i;
    }
    for (int i = 0; i < docSize; i++) {
        int tmp = sampleOrder[i];
        int swap = nextUnsignedInt(docSize);
        sampleOrder[i] = sampleOrder[swap];
        sampleOrder[swap] = tmp;
    }
    for (int i = 0; i < docSize; i++) {
        int n = sampleOrder[i];
        int k = topicAssignments_d[n];
        int v = docWords_d[n];
        docTopicCounts_d[k] -= 1;
        topicWordCounts[k][v] -= 1;
        topicWordSum[k] -= 1;
        double * topic_probs = probability_calculator(
                v, vbeta, docSize, numTopics, docTopicCounts_d, topicWordCounts,
                topicWordSum, alphas, hyperbeta, eta, var, response_d);
        k = sample_categorical(topic_probs, numTopics);
        free(topic_probs);
        topicAssignments_d[n] = k;
        docTopicCounts_d[k] += 1;
        topicWordCounts[k][v] += 1;
        topicWordSum[k] += 1;
    }
}

void gibbs_loop_helper(SamplerState * samplerState, double vbeta, int d,
        double * (* probability_calculator)(int, double, double, int, int *, int **, int *, double *, double, double *, double, double)) {
    int docSize = samplerState->corpusData->docSizes[d];
    int * topicAssignments_d = samplerState->topicAssignments[d];
    int * docWords_d = samplerState->corpusData->docWords[d];
    int * docTopicCounts_d = samplerState->docTopicCounts[d];
    double response_d = 0;
    if (d < samplerState->corpusData->numResponses) {
        response_d = samplerState->corpusData->responseValues[d];
    }
    gibbs_inner_loop(samplerState->numTopics, vbeta, docSize, topicAssignments_d,
            docWords_d, docTopicCounts_d, samplerState->topicWordCounts,
            samplerState->topicWordSum, samplerState->alphas, samplerState->hyperbeta,
            samplerState->eta, samplerState->var, response_d, probability_calculator);
}

/*
 * This function covers both supervised and semi-supervised learning.  It is
 * assumed that samplerState->topicAssignments, samplerState->docTopicCounts,
 * and samplerState->corpusData->responseValues are corresponding to each other
 * for the first samplerState->corpusData->numResponses elements (i.e., the
 * first items of each array correspond to the same document, the second items
 * of each array correspond to another document, etc.).
 *
 * To get supervised learning, make sure that samplerState->corpusData->numDocs
 * == samplerState->corpusData->numResponses.
 *
 * To get semi-supervised learning, make sure that
 * samplerState->corpusData->numDocs > samplerState->corpusData->numResponses.
 */
void gibbs_loop(SamplerState * samplerState, int loopNum) {
    double vbeta = samplerState->vocabSize * samplerState->hyperbeta;
    for (int i = 0; i < loopNum; i++) {
        for (int d = 0; d < samplerState->corpusData->numResponses; d++) {
            gibbs_loop_helper(samplerState, vbeta, d, sLDA_probability_calculator);
        }
        for (int d = samplerState->corpusData->numResponses; d < samplerState->corpusData->numDocs; d++) {
            gibbs_loop_helper(samplerState, vbeta, d, LDA_probability_calculator);
        }
    }
}

double dot(double * u, double * v, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += u[i] * v[i];
    }
    return result;
}

double calculate_R2(double * weights, int weightsCount, double * responseValues, int numResponses, double ** features) {
    double responseMean = 0.0;
    for (int i = 0; i < numResponses; i++) {
        responseMean += responseValues[i];
    }
    responseMean /= numResponses;
    double ss_res = 0.0;
    for (int i = 0; i < numResponses; i++) {
        double tmp = responseValues[i] - dot(weights, features[i], weightsCount);
        ss_res += tmp * tmp;
    }
    double ss_tot = 0.0;
    for (int i = 0; i < numResponses; i++) {
        double tmp = responseValues[i] - responseMean;
        ss_tot += tmp * tmp;
    }
    return 1 - (ss_res/ss_tot);
}

void trainPredictionWeights(SamplerState * samplerState) {
    // got help from http://www.netlib.org/lapack/lapacke.html under "Calling
    // DGELS"
    lapack_int info, m, n, lda, ldb, nrhs;

    m = samplerState->corpusData->numResponses;
    n = samplerState->numTopics;
    if (m < n) {
        // not enough info to do regression
        return;
    }
    nrhs = 1;
    lda = m;
    ldb = m;

    double ys[m];
    memcpy(ys, samplerState->corpusData->responseValues, m * sizeof(double));

    double ** zBars = malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        zBars[i] = malloc(n * sizeof(double));
        int sum = 0;
        for (int j = 0; j < n; j++) {
            sum += samplerState->docTopicCounts[i][j];
        }
        for (int j = 0; j < n; j++) {
            zBars[i][j] = ((double) samplerState->docTopicCounts[i][j]) / sum;
        }
    }
    double zBarsCopy[m*n];
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            zBarsCopy[i + (j * m)] = zBars[i][j];
            if (i == j) {
                // jitter to avoid singular matrices
                zBarsCopy[i + (j * m)] += (nextDouble() * 2e-10) - 1e-10;
            }
        }
    }

    info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', m, n, nrhs, zBarsCopy, lda, ys, ldb);
    if (info == 0) {
        // variance calculation according to Apache Math Commons' bias-corrected
        // algorithm (Blei and McAuliffe's version didn't give as good results)
        double predictions[samplerState->corpusData->numResponses];
        double totalsum = 0.0;
        for (int i = 0; i < samplerState->corpusData->numResponses; i++) {
            double sum = 0.0;
            for (int j = 0; j < samplerState->numTopics; j++) {
                sum += zBars[i][j] * ys[j];
            }
            totalsum += sum;
            predictions[i] = sum;
            double response_i = samplerState->corpusData->responseValues[i];
        }
        double mean = totalsum / samplerState->corpusData->numResponses;
        double accum = 0.0;
        double dev = 0.0;
        double accum2 = 0.0;
        const double numResponses = samplerState->corpusData->numResponses;
        for (int i = 0; i < numResponses; i++) {
            dev = predictions[i] - mean;
            accum += dev * dev;
            accum2 += dev;
        }
        samplerState->var = (accum - (accum2 * accum2 / numResponses)) / (numResponses - 1.0);
        memcpy(samplerState->eta, ys, samplerState->numTopics * sizeof(double));
        /*
        printf("%f\n", calculate_R2(samplerState->eta, samplerState->numTopics, samplerState->corpusData->responseValues,
                    samplerState->corpusData->numResponses, zBars));
        */
    } else {
        /*
        printf("LAPACK had an error: %d\n", info);
        printf("\tLAPACK_ROW_MAJOR N %d %d %d zBarsCopy %d ys %d\n", m, n, nrhs, lda, ldb);
        */
    }
    for (int i = 0; i < m; i++) {
        free(zBars[i]);
    }
    free(zBars);
}

SamplerState ** sample_sLDA(SamplerState * samplerState, int sampleCount, int * loopSchedule) {
    SamplerState ** result = malloc(sampleCount * sizeof(SamplerState *));
    for (int i = 0; i < sampleCount; i++) {
        gibbs_loop(samplerState, loopSchedule[i]);
        trainPredictionWeights(samplerState);
        result[i] = copySamplerState(samplerState);
    }
    return result;
}

CorpusData * buildPredictCorpus(int docSize, int * docWords) {
    CorpusData * result = malloc(sizeof(CorpusData));
    result->numDocs = 1;
    result->docSizes = malloc(sizeof(int *));
    result->docSizes[0] = docSize;
    result->docWords = malloc(sizeof(int *));
    const int docSizeIntArray = docSize * sizeof(int);
    result->docWords[0] = malloc(docSizeIntArray);
    memcpy(result->docWords[0], docWords, docSizeIntArray);
    result->numResponses = 0;
    result->responseValues = NULL;
    return result;
}

SamplerState * buildPredictSampler(SamplerState * curState, CorpusData * predictCorpus) {
    SamplerState * result = malloc(sizeof(SamplerState));
    const int numTopics = curState->numTopics;
    const int topicIntArrSize = numTopics * sizeof(int);
    const int topicDoubleArrSize = numTopics * sizeof(double);
    const int vocabSize = curState->vocabSize;
    const int vocabIntArrSize = vocabSize * sizeof(int);
    const int docSize = predictCorpus->docSizes[0];
    int * docWords = predictCorpus->docWords[0];
    result->numTopics = numTopics;
    result->vocabSize = vocabSize;
    result->alphas = malloc(topicDoubleArrSize);
    memcpy(result->alphas, curState->alphas, topicDoubleArrSize);
    result->hyperbeta = curState->hyperbeta;
    result->eta = malloc(topicDoubleArrSize);
    memcpy(result->eta, curState->eta, topicDoubleArrSize);
    result->var = curState->var;
    result->corpusData = copyCorpusData(predictCorpus);
    int * topicWordSum_aug = malloc(topicIntArrSize);
    memcpy(topicWordSum_aug, curState->topicWordSum, topicIntArrSize);
    int ** topicWordCounts_aug = malloc(numTopics * sizeof(int *));
    for (int j = 0; j < numTopics; j++) {
        topicWordCounts_aug[j] = malloc(vocabIntArrSize);
        memcpy(topicWordCounts_aug[j], curState->topicWordCounts[j], vocabIntArrSize);
    }
    int * docTopicCounts = malloc(topicIntArrSize);
    memset(docTopicCounts, 0, topicIntArrSize);
    int * docZs = malloc(docSize * sizeof(int));
    for (int j = 0; j < docSize; j++) {
        const int z = (int) (nextDouble() * numTopics);
        docZs[j] = z;
        docTopicCounts[z] += 1;
        topicWordSum_aug[z] += 1;
        topicWordCounts_aug[z][docWords[j]] += 1;
    }
    result->topicAssignments = malloc(sizeof(int *));
    result->topicAssignments[0] = docZs;
    result->docTopicCounts = malloc(sizeof(int *));
    result->docTopicCounts[0] = docTopicCounts;
    result->topicWordCounts = topicWordCounts_aug;
    result->topicWordSum = topicWordSum_aug;
    return result;
}

double * predict(int savedCount, SamplerState ** savedStates, int docSize,
        int * docWords, int numSamplesPerPredictChain, int * sampleSchedule) {
    const int totalPredictions = savedCount * numSamplesPerPredictChain;
    double * result = malloc(totalPredictions * sizeof(double));
    CorpusData * predictCorpus = buildPredictCorpus(docSize, docWords);
    for (int i = 0; i < savedCount; i++) {
        SamplerState * predictSampler = buildPredictSampler(savedStates[i], predictCorpus);
        const int numTopics = predictSampler->numTopics;
        double vbeta = predictSampler->vocabSize * predictSampler->hyperbeta;
        for (int j = 0; j < numSamplesPerPredictChain; j++) {
            gibbs_loop(predictSampler, sampleSchedule[j]);
            double zBars[numTopics];
            for (int k = 0; k < numTopics; k++) {
                zBars[k] = predictSampler->docTopicCounts[0][k] / (double) docSize;
            }
            double product = dot(predictSampler->eta, zBars, numTopics);
            result[(i*numSamplesPerPredictChain) + j] = product;
        }
        freeSamplerState(predictSampler);
    }
    freeCorpusData(predictCorpus);
    return result;
}

double * getExpectedTopicCounts(SamplerState * samplerState, int docSize, int * docWords, int numSamples, int * schedule) {
    CorpusData * doc = buildPredictCorpus(docSize, docWords);
    SamplerState * sampler = buildPredictSampler(samplerState, doc);
    const int numTopics = sampler->numTopics;
    double vbeta = sampler->vocabSize * sampler->hyperbeta;
    double * result = malloc(numTopics * sizeof(double));
    memset(result, 0, numTopics * sizeof(double));
    for (int i = 0; i < numSamples; i++) {
        gibbs_loop(sampler, schedule[i]);
        for (int j = 0; j < numTopics; j++) {
            result[j] += sampler->docTopicCounts[0][j];
        }
    }
    for (int i = 0; i < numTopics; i++) {
        result[i] /= docSize * numSamples;
    }
    return result;
}

void freeDoubleArray(double * a) {
    free(a);
}

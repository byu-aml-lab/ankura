import sys
import ankura
import sklearn
import random
import bs4
import scipy
import os.path
import pickle
import time
from sklearn.linear_model import LinearRegression
import numpy as np
import socket

# LABEL_NAME = 'coarse_newsgroup'
# LABEL_NAME = 'label'
LABEL_NAME = 'binary_rating'

def run_experiment(num_topics=50, label_weight=1, smoothing=0, epsilon=1e-5, train_size=10000, test_size=8000):

    # corpus = ankura.corpus.newsgroups() # LABEL_NAME = 'coarse_newsgroup'
    # corpus = ankura.corpus.tripadvisor() # LABEL_NAME = 'label'
    # corpus = ankura.corpus.amazon() # LABEL_NAME = 'binary_rating'
    corpus = ankura.corpus.yelp() # LABEL_NAME = 'binary_rating'

    total_time_start = time.time()

    train, test = ankura.pipeline.test_train_split(corpus, num_train=train_size, num_test=test_size, return_ids=True)

    Q, labels = ankura.anchor.build_labeled_cooccurrence(corpus, LABEL_NAME, set(train[0]), label_weight, smoothing)

    anchors = ankura.anchor.gram_schmidt_anchors(corpus, Q, num_topics)

    anchor_start = time.time()
    C, topics = ankura.anchor.recover_topics(Q, anchors, get_c=True)
    anchor_end = time.time()

    anchor_time = anchor_end - anchor_start

    classifier = ankura.topic.free_classifier_dream(corpus, LABEL_NAME, set(train[0]), topics, C, labels)

    contingency = ankura.validate.Contingency()
    for i, doc in enumerate(test[1].documents):
        contingency[doc.metadata[LABEL_NAME], classifier(doc)] += 1

    total_time_end = time.time()
    total_time = total_time_end - total_time_start


    return contingency.accuracy(), total_time




if __name__ == "__main__":

    n_runs = 30
    epsilon = 1e-4
    spacer = '{:<12}'

    potatoes = {'0potato':[1,0,[10,20]], '24potato':[1,1e-4,[10,20]], '2potato':[1,1e-1,[10,20]],
                    '3potato':[1,0,[50,100]], '4potato':[1,1e-4,[50,100]], '5potato':[1,1e-1,[50,100]],
                    '6potato':[10,0,[10,20]], '7potato':[10,1e-4,[10,20]], '8potato':[10,1e-1,[10,20]],
                    '9potato':[10,0,[50,100]], '10potato':[10,1e-4,[50,100]], '11potato':[10,1e-1,[50,100]],
                    '12potato':[1000,0,[10,20]], '13potato':[1000,1e-4,[10,20]], '14potato':[1000,1e-1,[10,20]],
                    '21potato':[1000,0,[50,100]], '16potato':[1000,1e-4,[50,100]], '17potato':[1000,1e-1,[50,100]]}

    for p in potatoes:
        if socket.gethostname() == p:
            print((spacer + spacer + spacer + spacer + spacer).format("Topic", "Weight", "Smoothing", "Accuracy", "Time"))

            weight = potatoes[p][0]
            smoothing = potatoes[p][1]
            topics = potatoes[p][2]

            for topicNum in topics:
                accuracy = []
                times = []

                for _ in range(n_runs):
                    current_accuracy, current_time = run_experiment(topicNum, weight, smoothing, epsilon)
                    accuracy.append(current_accuracy)
                    times.append(current_time)

                print((spacer + spacer + spacer + spacer + spacer).format(topicNum, weight, smoothing, round(np.mean(accuracy), 4)*100, round(np.mean(times), 4)))

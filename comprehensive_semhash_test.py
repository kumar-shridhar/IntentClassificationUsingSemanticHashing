from __future__ import unicode_literals

import codecs
import csv
import json
import math
import os
import random
import re
import sys
from itertools import product
from time import time

from sklearn import metrics, model_selection
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer,
                                             TfidfVectorizer)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier,
                                  RidgeClassifier, SGDClassifier)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density

from tqdm import tqdm
import numpy as np
import spacy

from MeraDataset import MeraDataset
from utils import semhash_corpus, preprocess, tokenize, trim

# ## Benchmarking using SemHash on NLU Evaluation Corpora
# 
# This notebook benchmarks the results on the 3 NLU Evaluation Corpora:
# 1. Ask Ubuntu Corpus
# 2. Chatbot Corpus
# 3. Web Application Corpus
# 
# 
# More information about the dataset is available here: 
# 
# https://github.com/sebischair/NLU-Evaluation-Corpora
# 
# 
# * Semantic Hashing is used as a featurizer. The idea is taken from the paper:
# 
# https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/
# 
# * Benchmarks are performed on the same train and test datasets used by the other benchmarks performed in the past. One important paper that benchmarks the datasets mentioned above on some important platforms (Dialogflow, Luis, Watson and RASA) is : 
# 
# http://workshop.colips.org/wochat/@sigdial2017/documents/SIGDIAL22.pdf
# 
# * Furthermore, Botfuel made another benchmarks with more platforms (Recast, Snips and their own) and results can be found here: 
# 
# https://github.com/Botfuel/benchmark-nlp-2018
# 
# * The blogposts about the benchmarks done in the past are available at : 
# 
# https://medium.com/botfuel/benchmarking-intent-classification-services-june-2018-eb8684a1e55f
# 
# https://medium.com/snips-ai/an-introduction-to-snips-nlu-the-open-source-library-behind-snips-embedded-voice-platform-b12b1a60a41a
# 
# * To be very fair on our benchmarks and results, we used the same train and test set used by the other benchmarks and no cross validation or stratified splits were used. The test data was not used in any way to improve the results. The dataset used can be found here:
# 
# https://github.com/Botfuel/benchmark-nlp-2018/tree/master/results
# 
# 

# Spacy english dataset with vectors needs to be present. It can be downloaded using the following command:
# !pip3 install spacy
# python -m spacy download en_core_web_lg
nlp=spacy.load('en_core_web_lg')
print('Running')

RESULT_FILE = "result5.csv"
METADATA_FILE = "metadata5.csv"
NUMBER_OF_RUNS_PER_SETTING = 10


class SemhashTest():
    def __init__(
        self,
        benchmark_dataset: str,
        vectorizer: str,
        mistake_distance = -1,
        additional_synonyms = -1,
        additional_augments = -1,
        oversampling = False,
        synonym_extra_samples = False,
        augment_extra_samples = False
        ):
        self.benchmark_dataset = benchmark_dataset              # Choose from 'AskUbuntu', 'Chatbot' or 'WebApplication'
        self.vectorizer_name = vectorizer                       # which vectorizer to use. choose between "count", "hash", and "tfidf"

        self.filename_train = "datasets/KL/" + benchmark_dataset + "/train.csv"
        self.filename_test = "datasets/KL/" + benchmark_dataset + "/test.csv"

        if self.benchmark_dataset == "Chatbot":
            self.intent_dict = {"DepartureTime":0, "FindConnection":1}
        elif self.benchmark_dataset == "AskUbuntu":
            self.intent_dict = {"Make Update":0, "Setup Printer":1, "Shutdown Computer":2, "Software Recommendation":3, "None":4}
        elif self.benchmark_dataset == "WebApplication":
            self.intent_dict = {"Download Video":0, "Change Password":1, "None":2, "Export Data":3, "Sync Accounts":4,
                        "Filter Spam":5, "Find Alternative":6, "Delete Account":7}
        else:
            self.indent_dict = None

        self.train_mera_dataset = MeraDataset(
            self.filename_train, 
            additional_synonyms=additional_synonyms,
            additional_augments=additional_augments,
            oversampling=oversample,
            synonym_extra_samples=synonym_extra_samples,
            augment_extra_samples=augment_extra_samples,
            mistake_distance=mistake_distance
        )

        if self.benchmark_dataset == "Chatbot":
            self.target_names = ["Departure Time", "Find Connection"]
        elif self.benchmark_dataset == "AskUbuntu":
            self.target_names = ["Make Update", "Setup Printer", "Shutdown Computer","Software Recommendation", "None"]
        elif self.benchmark_dataset == "WebApplication":
            self.target_names = ["Download Video", "Change Password", "None", "Export Data", "Sync Accounts",
                      "Filter Spam", "Find Alternative", "Delete Account"]
        else:
            self.target_names = None

    def read_CSV_datafile(self, filename):
        X = []
        y = []
        with open(filename,'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                X.append(row[0])
                y.append(self.intent_dict[row[1]])
        return X, y

    def get_train_set(self):
        return self.read_CSV_datafile(self.filename_train)

    def get_test_set(self):
        return self.read_CSV_datafile(self.filename_test)
    
    def get_intent(self):
        return self.intent_dict

    def get_target_names(self):
        return self.target_names

    def get_vectorizer(self, corpus):
        if self.vectorizer_name == "count":
            vectorizer = CountVectorizer(analyzer='word')#,ngram_range=(1,1))
            vectorizer.fit(corpus)
            feature_names = vectorizer.get_feature_names()
        elif self.vectorizer_name == "hash":
            vectorizer = HashingVectorizer(analyzer='word', n_features=2**10)
            vectorizer.fit(corpus)
            feature_names = None
        elif self.vectorizer_name == "tfidf":
            vectorizer = TfidfVectorizer(analyzer='word')
            vectorizer.fit(corpus)
            feature_names = vectorizer.get_feature_names()
        else:
            raise Exception("{} is not a recognized Vectorizer".format(self.vectorizer_name))
        return vectorizer, feature_names


    def data_for_training(self, preprocessor, tokenizer):
        vectorizer, feature_names = self.get_vectorizer(X_train_raw)

        X_train = vectorizer.transform(X_train_raw).toarray()
        X_test = vectorizer.transform(X_test_raw).toarray()

        return X_train, y_train_raw, X_test, y_test_raw, feature_names



#Comprehensive settings testing
comprehensive_testing_space = product(
    ['AskUbuntu', 'Chatbot', 'WebApplication'],
    [
        (False, False, False),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True)
    ],
    [0,4],
    [0,4],
    [2.1],
    ["tfidf", "hash", "count"]
    )

orignal_paper_space = product(
    ['AskUbuntu', 'Chatbot', 'WebApplication'],
    [(True, True, False)],
    [0],
    [0],
    [2.1],
    ["tfidf"]
    )

USE_ORIGNAL_PAPER_SETTING = True
search_space = orignal_paper_space if USE_ORIGNAL_PAPER_SETTING else comprehensive_testing_space


#Settings from the original paper
for benchmark_dataset, (oversample, synonym_extra_samples, augment_extra_samples), additional_synonyms, additional_augments, mistake_distance, VECTORIZER in search_space:

    semHashTest = SemhashTest(
        benchmark_dataset,
        VECTORIZER,
        additional_synonyms=additional_synonyms,
        additional_augments=additional_augments,
        oversampling=oversample,
        synonym_extra_samples=synonym_extra_samples,
        augment_extra_samples=augment_extra_samples
        )

    print("Using dataset available at:", "./datasets/KL/" + benchmark_dataset + "/train.csv")
    t0 = time()
    splits = semHashTest.train_mera_dataset.get_splits()
    xS_train = []
    yS_train = []
    for elem in splits[0]["train"]["X"]:
        xS_train.append(elem)
    print(xS_train[:5])
    intent_dict = semHashTest.get_intent()
    for elem in splits[0]["train"]["y"]:
        yS_train.append(intent_dict[elem])
    preprocess_time = time()-t0
    print(len(xS_train))

    X_train_raw, y_train_raw = semHashTest.get_train_set()
    X_test_raw, y_test_raw = semHashTest.get_test_set()
    print(y_train_raw[:5])
    print(X_test_raw[:5])
    print(y_test_raw[:5])
    X_train_raw = xS_train
    y_train_raw = yS_train

    print("Training data samples: \n",X_train_raw, "\n\n")

    print("Class Labels: \n", y_train_raw, "\n\n")

    print("Size of Training Data: {}".format(len(X_train_raw)))


    t0 = time()
    X_train_raw = semhash_corpus(X_train_raw)
    X_test_raw = semhash_corpus(X_test_raw)
    semhash_time = time()-t0

    print(X_train_raw[:5])
    print(y_train_raw[:5])
    print()
    print(X_test_raw[:5])
    print(y_test_raw[:5])




    # #############################################################################
    # Benchmark classifiers
    def benchmark(clf, X_train, y_train, X_test, y_test, target_names,
                  print_report=True, feature_names=None, print_top10=False,
                  print_cm=True):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        f1_score = metrics.f1_score(y_test, pred, average='weighted')

        #bad_pred = X_test[pred != y_test]

        print("accuracy:   %0.3f" % score)
        #print("Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            if print_top10 and feature_names is not None:
                print("top 10 keywords per class:")
                for i, label in enumerate(["Make Update", "Setup Printer", "Shutdown Computer","Software Recommendation", "None"]):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print(trim("%s: %s" % (label, " ".join([feature_names[i] for i in top10]))))
            print()

        if print_report:
            print("classification report:")
            print(metrics.classification_report(y_test, pred,labels = range(len(target_names)),
                                                target_names=target_names))

        if print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]

        with open("./" + RESULT_FILE, 'a', encoding='utf8') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter='\t')
            fileWriter.writerow([benchmark_dataset,str(clf),str(oversample),str(synonym_extra_samples),str(augment_extra_samples),
                                 str(additional_synonyms),str(additional_augments), str(mistake_distance), str(score), str(f1_score), str(train_time), str(test_time)])

        return clf_descr, score, train_time, test_time, f1_score


    t0 = time()
    X_train, y_train, X_test, y_test, feature_names = semHashTest.data_for_training(preprocess, tokenize)
    vectorize_time = time() - t0

    with open("./" + METADATA_FILE, 'a', encoding='utf8') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter='\t')
            fileWriter.writerow([benchmark_dataset,str(oversample),str(synonym_extra_samples),str(augment_extra_samples),str(additional_synonyms),str(additional_augments),str(mistake_distance),str(preprocess_time),str(semhash_time),str(vectorize_time)])


    print(X_train[0].tolist())
    print(y_train[0])
    print(feature_names)


    for _ in enumerate(range(NUMBER_OF_RUNS_PER_SETTING)):
        i_s = 0
        split = 0
        print("Evaluating Split {}".format(i_s))
        target_names = semHashTest.get_target_names()

        print("Train Size: {}\nTest Size: {}".format(X_train.shape[0], X_test.shape[0]))
        results = []
        
        #alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
        parameters_mlp = {'hidden_layer_sizes': [(100,50), (300, 100),(300,200,100)]}
        parameters_RF = { "n_estimators": [50,60,70], "min_samples_leaf" : [1, 11]}

        k_range = list(range(3,7))
        parameters_knn = {'n_neighbors': k_range}

        for clf, name in [  
                (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                (GridSearchCV(KNeighborsClassifier(n_neighbors=5), parameters_knn, cv=5),"gridsearchknn"),
                #(Perceptron(n_iter=50), "Perceptron"),
                (GridSearchCV(MLPClassifier(activation='tanh'), parameters_mlp, cv=5), "gridsearchmlp"),
                (PassiveAggressiveClassifier(), "Passive-Aggressive"),
                (GridSearchCV(RandomForestClassifier(n_estimators=10), parameters_RF, cv=5), "gridsearchRF"),
                (SGDClassifier(alpha=.0001, penalty="elasticnet"), "sgdelasticnet"),
                (NearestCentroid(), "nearestcentroid"),
                (MultinomialNB(alpha=.01), "multinomialNB"),
                (BernoulliNB(alpha=.01), "binomialNB"),
                (KMeans(n_clusters=2, init='k-means++', max_iter=300, verbose=0, random_state=0, tol=1e-4), "kmeans2cluster"),
                (LogisticRegression(C=1.0, class_weight=None, dual=False,
                                    fit_intercept=True, intercept_scaling=1, max_iter=100,
                                    multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
                                    solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
                    "logreressioin"),
                (Pipeline([
                                    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                                                    tol=1e-3))),
                                    ('classification', LinearSVC(penalty="l2"))]), "pipelinelinearsvc")

        ]:

            print('=' * 80)
            print(name)
            result = benchmark(clf, X_train, y_train, X_test, y_test, target_names,
                                     feature_names=feature_names)
            results.append(result)

        parameters_Linearsvc = [{'C': [1, 10], 'gamma': [0.1,1.0]}]
        for penalty in ["l2", "l1"]:
            for clf, name in [  
                    (LinearSVC(penalty=penalty, dual=False,tol=1e-3), "linearsvc"),
                    (SGDClassifier(alpha=.0001, penalty=penalty),"sgdclassifier"),
                    #(Perceptron(n_iter=50), "Perceptron"),
                    (GridSearchCV(MLPClassifier(activation='tanh'), parameters_mlp, cv=5), "gridsearchmlp"),
                    (PassiveAggressiveClassifier(), "Passive-Aggressive"),
                    (GridSearchCV(RandomForestClassifier(n_estimators=10), parameters_RF, cv=5), "gridsearchRF"),
            ]:
                print("="*80)
                print(penalty, ": ", name)
                result = benchmark(clf, X_train, y_train, X_test, y_test, target_names,
                                        feature_names=feature_names)
                results.append(result)

    print(len(X_train))
































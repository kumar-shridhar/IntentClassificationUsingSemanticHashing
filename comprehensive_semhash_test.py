from __future__ import unicode_literals

import sys
import re
import os
import codecs
import json
import csv
import spacy
import sklearn
import math
import random
import numpy as np
from time import time
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from collections import OrderedDict
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from MeraDataset import MeraDataset
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



# import os
# os.environ['LDFLAGS'] = '-framework CoreFoundation -framework SystemConfiguration'
# !pip3 install spacy

#coding: utf-8
# import locale
# print(locale.getlocale())

# Spacy english dataset with vectors needs to be present. It can be downloaded using the following command:
# 
# python -m spacy download en_core_web_lg
nlp=spacy.load('en_core_web_lg')
print('\nSpacy en core loaded!\n')


"""
Get nouns and verbs list from WordNet (just another NLTK corpus reader)
Synsets are a set of synonyms that share a common meaning
"""
nouns = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('n')}
verbs = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('v')}

# for hyper_bench in ['AskUbuntu', 'Chatbot', 'WebApplication']:
#     benchmark_dataset = hyper

#     for hyper_over in [True, False]:
#         oversample = hyper_over

#         for hyper_syn_extra in [True, False]:
#             synonym_extra_samples = hyper_syn_extra

#             for hyper_aug in [True, False]:
#                 augm
#Hyperparameters
benchmark_dataset = '' # Choose from 'AskUbuntu', 'Chatbot' or 'WebApplication'
oversample = False             # Whether to oversample small classes or not. True in the paper
synonym_extra_samples = False  # Whether to replace words by synonyms in the oversampled samples. True in the paper
augment_extra_samples = False # Whether to add random spelling mistakes in the oversampled samples. False in the paper
additional_synonyms = -1      # How many extra synonym augmented sentences to add for each sentence. 0 in the paper
additional_augments = -1       # How many extra spelling mistake augmented sentences to add for each sentence. 0 in the paper
mistake_distance = -1        # How far away on the keyboard a mistake can be
VECTORIZER = "tfidf"                 #which vectorizer to use. choose between "count", "hash", and "tfidf"

RESULT_FILE = "result5.csv"
METADATA_FILE = "metadata5.csv"
NUMBER_OF_RUNS_PER_SETTING = 10

#Comprehensive settings testing
#for benchmark_dataset, (oversample, synonym_extra_samples, augment_extra_samples), additional_synonyms, additional_augments, mistake_distance, VECTORIZER in product(['AskUbuntu', 'Chatbot', 'WebApplication'], [(False, False, False),(True, False, False),(True, False, True),(True, True, False),(True, True, True)], [0,4], [0,4], [2.1], ["tfidf", "hash", "count"]):


class Trainingstuff:

    def __init__(self):
        self.intent_dict = {}

    def read_CSV_datafile(self, filename):
        """Process CSV file and return dataset with labels
        
        Returns: X,y
        """    
        X = []
        y = []
        with open(filename,'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                X.append(row[0])
                if benchmark_dataset == 'AskUbuntu':
                    y.append(self.intent_dict[row[1]])
                elif benchmark_dataset == 'Chatbot':
                    y.append(self.intent_dict[row[1]])
                else:
                    y.append(self.intent_dict[row[1]])           
        return X,y
    
    @staticmethod
    def tokenize(doc):
        """Segment text into words, punctuations marks etc.
        
        Returns: List of strings containing each token in `sentence`
        """
        #return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
        #                            doc) if i != '' and i != ' ' and i != '\n']
        tokens = []
        doc = nlp.tokenizer(doc)
        for token in doc:
            tokens.append(token.text)
        return tokens
    
    @staticmethod
    def preprocess(doc):
        """Stopwords are a list of most common words of a language 
        that are often useful to filter out (eg. "and", "I")

        Returns: Sentences without stopwords 
        """
        clean_tokens = []
        doc = nlp(doc)
        for token in doc:
            if not token.is_stop:
                clean_tokens.append(token.lemma_)
        return " ".join(clean_tokens)
    
    def find_ngrams(self, input_list, n):
        return zip(*[input_list[i:] for i in range(n)])
    
    def semhash_tokenizer(self, text):
        tokens = text.split(" ")
        final_tokens = []
        for unhashed_token in tokens:
            hashed_token = "#{}#".format(unhashed_token)
            final_tokens += [''.join(gram)
                               for gram in list(self.find_ngrams(list(hashed_token), 3))]
        return final_tokens

    def semhash_corpus(self, corpus):
        new_corpus = []
        for sentence in corpus:
            sentence = self.preprocess(sentence)
            tokens = self.semhash_tokenizer(sentence)
            new_corpus.append(" ".join(map(str,tokens)))
        return new_corpus

    def get_vectorizer(self, corpus, preprocessor=None, tokenizer=None):
        """CountVectorizer: Convert a collection of text documents to a matrix of token counts
        
        HashingVectorizer: Convert a collection of text documents to a matrix of token occurrences
        
        TfidfVectorizer: Convert a collection of raw documents to a matrix of TF-IDF features.
        """
        if VECTORIZER == "count":
            vectorizer = CountVectorizer(analyzer='word')#,ngram_range=(1,1))
            vectorizer.fit(corpus)
            feature_names = vectorizer.get_feature_names()
        elif VECTORIZER == "hash":
            vectorizer = HashingVectorizer(analyzer='word', n_features=2**10) #non_negative=True
            vectorizer.fit(corpus)
            feature_names = None
        elif VECTORIZER == "tfidf":
            vectorizer = TfidfVectorizer(analyzer='word')
            vectorizer.fit(corpus)
            feature_names = vectorizer.get_feature_names()
        else:
            raise Exception("{} is not a recognized Vectorizer".format(VECTORIZER))
        return vectorizer, feature_names

    def trim(self, s):
        """Trim string to fit on terminal (assuming 80-column display)
        """
        return s if len(s) <= 80 else s[:77] + "..."

    def plot_results(self, results):
        """Make some plots of results
        """
        indices = np.arange(len(results))
        results = [[x[i] for x in results] for i in range(4)]
        clf_names, score, training_time, test_time = results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)
        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="score", color='navy')
        plt.barh(indices + .3, training_time, .2, label="training time",
                    color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)
        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        plt.show()
    

    ##############################################################################
    # Benchmark classifiers
    def benchmark(self, clf, X_train, y_train, X_test, y_test, target_names,
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
                    print(self.trim("%s: %s" % (label, " ".join([feature_names[i] for i in top10]))))
            print()

        if print_report:
            print("classification report:")
            print(metrics.classification_report(y_test, pred,labels = range(len(target_names)), target_names=target_names))

        if print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))

        with open("./"+RESULT_FILE, 'a', encoding='utf8') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter='\t')
            fileWriter.writerow([benchmark_dataset,str(clf),str(oversample),str(synonym_extra_samples),str(augment_extra_samples),str(additional_synonyms),str(additional_augments), str(mistake_distance), str(score), str(f1_score), str(train_time), str(test_time)])

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time, f1_score


    def start_training(self):
        #Settings from the original paper
        for benchmark_dataset, (oversample, synonym_extra_samples, augment_extra_samples), additional_synonyms, additional_augments, mistake_distance, VECTORIZER in product(['AskUbuntu', 'Chatbot', 'WebApplication'], [(True, True, False)], [0], [0], [2.1], ["tfidf"]):

            if benchmark_dataset == "Chatbot":
                self.intent_dict = {"DepartureTime":0, "FindConnection":1}
            elif benchmark_dataset == "AskUbuntu":
                self.intent_dict = {"Make Update":0, "Setup Printer":1, "Shutdown Computer":2, "Software Recommendation":3, "None":4}
            elif benchmark_dataset == "WebApplication":
                self.intent_dict = {"Download Video":0, "Change Password":1, "None":2, "Export Data":3, "Sync Accounts":4,
                            "Filter Spam":5, "Find Alternative":6, "Delete Account":7}

            filename_train = "datasets/KL/" + benchmark_dataset + "/train.csv"
            filename_test = "datasets/KL/" + benchmark_dataset + "/test.csv"

            print("datasets/KL/" + benchmark_dataset + "/train.csv")
            t0 = time()
            dataset = MeraDataset("datasets/KL/" + benchmark_dataset + "/train.csv", mistake_distance=mistake_distance, nouns = nouns, verbs=verbs, additional_synonyms=additional_synonyms, additional_augments=additional_augments, synonym_extra_samples=synonym_extra_samples, augment_extra_samples=augment_extra_samples, oversample=oversample)
            
            print("mera****************************")
            splits = dataset.get_splits()
            xS_train = []
            yS_train = []
            for elem in splits[0]["train"]["X"]:
                xS_train.append(elem)
            print(xS_train[:5])

            for elem in splits[0]["train"]["y"]:
                yS_train.append(self.intent_dict[elem])
            preprocess_time = time()-t0
            print(len(xS_train))
            
            X_train_raw, y_train_raw = self.read_CSV_datafile(filename = filename_train)
            X_test_raw, y_test_raw = self.read_CSV_datafile(filename = filename_test)
            print(y_train_raw[:5])
            print(X_test_raw[:5])
            print(y_test_raw[:5])
            X_train_raw = xS_train
            y_train_raw = yS_train

            print("Training data samples: \n",X_train_raw, "\n\n")

            print("Class Labels: \n", y_train_raw, "\n\n")

            print("Size of Training Data: {}".format(len(X_train_raw)))

            # SemHash

            t0 = time()
            X_train_raw = self.semhash_corpus(X_train_raw)
            X_test_raw = self.semhash_corpus(X_test_raw)
            semhash_time = time()-t0


            print(X_train_raw[:5])
            print(y_train_raw[:5])
            print()
            print(X_test_raw[:5])
            print(y_test_raw[:5])


            def data_for_training():
                vectorizer, feature_names = self.get_vectorizer(X_train_raw, preprocessor=self.preprocess, tokenizer=self.tokenize)
                X_train = vectorizer.transform(X_train_raw).toarray()
                X_test = vectorizer.transform(X_test_raw).toarray()
                return X_train, y_train_raw, X_test, y_test_raw, feature_names
            

            t0 = time()
            X_train, y_train, X_test, y_test, feature_names = data_for_training()
            vectorize_time = time()-t0

            with open("./"+METADATA_FILE, 'a', encoding='utf8') as csvFile:
                    fileWriter = csv.writer(csvFile, delimiter='\t')
                    fileWriter.writerow([benchmark_dataset,str(oversample),str(synonym_extra_samples),str(augment_extra_samples),str(additional_synonyms),str(additional_augments),str(mistake_distance),str(preprocess_time),str(semhash_time),str(vectorize_time)])


            print(X_train[0].tolist())
            print(y_train[0])
            print(feature_names)

            for _ in enumerate(range(NUMBER_OF_RUNS_PER_SETTING)):
                i_s = 0
                split = 0
                print("Evaluating Split {}".format(i_s))
                target_names = None
                if benchmark_dataset == "Chatbot":
                    target_names = ["Departure Time", "Find Connection"]
                elif benchmark_dataset == "AskUbuntu":
                    target_names = ["Make Update", "Setup Printer", "Shutdown Computer","Software Recommendation", "None"]
                elif benchmark_dataset == "WebApplication":
                    target_names = ["Download Video", "Change Password", "None", "Export Data", "Sync Accounts",
                            "Filter Spam", "Find Alternative", "Delete Account"]

                print("Train Size: {}\nTest Size: {}".format(X_train.shape[0], X_test.shape[0]))
                results = []

                #alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
                parameters_mlp={'hidden_layer_sizes':[(100,50), (300, 100),(300,200,100)]}
                parameters_RF={ "n_estimators" : [50,60,70],"min_samples_leaf" : [1, 11]}
                
                k_range = list(range(3,7))
                parameters_knn = {'n_neighbors':k_range}
                for clf, name in [  
                        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                        (GridSearchCV(KNeighborsClassifier(n_neighbors=5),parameters_knn, cv=5),"gridsearchknn"),
                        #(Perceptron(n_iter=50), "Perceptron"),
                        (GridSearchCV(MLPClassifier(activation='tanh'),parameters_mlp, cv=5),"gridsearchmlp"),
                        (PassiveAggressiveClassifier(max_iter=100), "Passive-Aggressive"),
                        (GridSearchCV(RandomForestClassifier(n_estimators=10),parameters_RF, cv=5),"gridsearchRF"),
                        (SGDClassifier(alpha=.0001, max_iter=100,penalty="elasticnet"),"elasticsgd"),
                        (NearestCentroid(),"nearestcentroid"),
                        (MultinomialNB(alpha=.01),"naivebayes"),
                        (BernoulliNB(alpha=.01), "bernoullinb"),
                        (LogisticRegression(C=1.0, class_weight=None, dual=False,fit_intercept=True, 
                                            intercept_scaling=1, max_iter=100,multi_class='ovr', 
                                            n_jobs=1, penalty='l2', random_state=None, solver='liblinear', 
                                            tol=0.0001, verbose=0, warm_start=False), "lr"),
                        (Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,tol=1e-3))),
                                   ('classification', LinearSVC(penalty="l2"))]), "svcpipeline"),
                        (KMeans(n_clusters=2, init='k-means++', max_iter=300,
                            verbose=0, random_state=0, tol=1e-4), "kmeans")
                ]:

                    print('=' * 80)
                    print(name)
                    result = self.benchmark(clf, X_train, y_train, X_test, y_test, target_names, feature_names=feature_names)
                    results.append(result)

                for penalty in ["l2", "l1"]:
                    result = self.benchmark(LinearSVC(penalty=penalty, dual=False,tol=1e-3),
                                            X_train, y_train, X_test, y_test, target_names,
                                            feature_names=feature_names)
                    results.append(result)

                    result = self.benchmark(SGDClassifier(alpha=.0001, max_iter=100,
                                                        penalty=penalty),
                                            X_train, y_train, X_test, y_test, target_names,
                                            feature_names=feature_names)
                    results.append(result)
                
                #plot_results(results)

            print(len(X_train))



def main():
    print("\nPath:\n", sys.path)
    print(MeraDataset.get_synonyms("search",-1))
    new_test = Trainingstuff()
    new_test.start_training()
    print("\nTraining Finished!\n")


if __name__== "__main__":
    main()














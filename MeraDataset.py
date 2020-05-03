import numpy as np
import csv
from nltk.corpus import wordnet
import random
import math
import spacy
from tqdm import tqdm
from collections import OrderedDict
import spacy

nlp = spacy.load('en_core_web_sm')

class MeraDataset():
    """ Class to find typos based on the keyboard distribution, for QWERTY style keyboards
    It's the actual test set as defined in the paper that we comparing against.
    """

    def __init__(self, dataset_path, mistake_distance, nouns, verbs, 
            additional_synonyms, additional_augments, synonym_extra_samples, 
            augment_extra_samples, oversample):
        """ Instantiate the object.
        
        @Param dataset_path: The directory which contains the data set
        """
        self.oversample = oversample
        self.augment_extra_samples = augment_extra_samples
        self.synonym_extra_samples = synonym_extra_samples
        self.additional_augments = additional_augments
        self.additional_synonyms = additional_synonyms
        self.verbs = verbs
        self.nouns = nouns
        self.mistake_distance=mistake_distance
        self.dataset_path = dataset_path
        self.X_test, self.y_test, self.X_train, self.y_train = self.load()
        self.keyboard_cartesian = {'q': {'x': 0, 'y': 0}, 'w': {'x': 1, 'y': 0}, 'e': {'x': 2, 'y': 0},
                                    'r': {'x': 3, 'y': 0}, 't': {'x': 4, 'y': 0}, 'y': {'x': 5, 'y': 0},
                                    'u': {'x': 6, 'y': 0}, 'i': {'x': 7, 'y': 0}, 'o': {'x': 8, 'y': 0},
                                    'p': {'x': 9, 'y': 0}, 'a': {'x': 0, 'y': 1}, 'z': {'x': 0, 'y': 2},
                                    's': {'x': 1, 'y': 1}, 'x': {'x': 1, 'y': 2}, 'd': {'x': 2, 'y': 1},
                                    'c': {'x': 2, 'y': 2}, 'f': {'x': 3, 'y': 1}, 'b': {'x': 4, 'y': 2},
                                    'm': {'x': 6, 'y': 2}, 'j': {'x': 6, 'y': 1}, 'g': {'x': 4, 'y': 1},
                                    'h': {'x': 5, 'y': 1}, 'k': {'x': 7, 'y': 1}, 'ö': {'x': 11,'y': 0},
                                    'l': {'x': 8, 'y': 1}, 'v': {'x': 3, 'y': 2}, 'n': {'x': 5, 'y': 2},
                                    'ß': {'x': 10,'y': 2}, 'ü': {'x': 10,'y': 2}, 'ä': {'x': 10,'y': 0}}
        self.nearest_to_i = self.get_nearest_to_i(self.keyboard_cartesian)
        self.splits = self.stratified_split()


    def get_nearest_to_i(self, keyboard_cartesian):
        """ Get the nearest key to the one read

        @Params: keyboard_cartesian The layout of the QWERTY keyboard for English
                
        Returns: dictionary of eaculidean distances for the characters
        """
        nearest_to_i = {}
        for i in keyboard_cartesian.keys():
            nearest_to_i[i] = []
            for j in keyboard_cartesian.keys():
                if self._euclidean_distance(i, j) < self.mistake_distance: #was > 1.2
                    nearest_to_i[i].append(j)
        return nearest_to_i

    def preprocess(self, doc):
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

    def _shuffle_word(self, word, cutoff=0.7):
        """ Rearange the given characters in a word simulating typos given a probability.

        @Param word: A single word coming from a sentence

        @Param cutoff: The cutoff probability to make a change (default 0.9)

        @Returns: The word rearranged 
        """
        word = list(word.lower())
        if random.uniform(0, 1.0) > cutoff:
            loc = np.random.randint(0, len(word))
            if word[loc] in self.keyboard_cartesian:
                word[loc] = random.choice(self.nearest_to_i[word[loc]])
        return ''.join(word)        

    def _euclidean_distance(self, a, b):
        """ Calculates the euclidean between 2 points in the keyboard
        
        @Param a: Point one 

        @Param b: Point two
        
        Returns: The euclidean distance between the two points
        """
        X = (self.keyboard_cartesian[a]['x'] - self.keyboard_cartesian[b]['x']) ** 2
        Y = (self.keyboard_cartesian[a]['y'] - self.keyboard_cartesian[b]['y']) ** 2
        return math.sqrt(X + Y)

    def _get_augment_sentence(self, sentence):
        return ' '.join([self._shuffle_word(item) for item in sentence.split(' ')])

    def _augment_sentence(self, sentence, num_samples):
        """ Augment the dataset of file with a sentence shuffled

        @Param sentence: The sentence from the set

        @Param num_samples: The number of sentences to genererate
        
        Returns: A set of augmented sentences
        """
        sentences = []
        for _ in range(num_samples):
            sentences.append(self._get_augment_sentence(sentence))
        sentences = list(set(sentences))
        # print("sentences", sentences)
        return sentences + [sentence]

    def _augment_split(self, X_train, y_train, num_samples=100):
        """Split the augmented train dataset

        @Param X_train: The full array of sentences

        @Param y_train: The train labels in the train dataset

        @Param num_samples: the number of new sentences to create (default 1000)

        Returns: Augmented training dataset
        """
        Xs, ys = [], []
        for X, y in zip(X_train, y_train):
            tmp_x = self._augment_sentence(X, num_samples)
            sample = [[Xs.append(item), ys.append(y)] for item in tmp_x]
            #print(X, y)
            #print(self.augmentedFile+str(self.nSamples)+".csv")

        with open("./datasets/KL/Chatbot/train_augmented.csv", 'w', encoding='utf8') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter='\t')
            for i in range(0, len(Xs)-1):
                fileWriter.writerow([Xs[i] + '\t' + ys[i]])
                # print(Xs[i], "\t", ys[i])
                # print(Xs[i])
            # fileWriter.writerows(Xs + ['\t'] + ys)
        return Xs, ys

    def _synonym_word(self, word, cutoff=0.5):
        """Randomly replaces the nouns and verbs by synonyms
        """
        if random.uniform(0, 1.0) > cutoff and len(self.get_synonyms(word)) > 0 and word in self.nouns and word in self.verbs:
            return random.choice(self.get_synonyms(word))
        return word

    def _get_synonym_sentence(self, sentence, cutoff = 0.5):
        """Randomly replace words (nouns and verbs) in sentence by synonyms
        """
        return ' '.join([self._synonym_word(item, cutoff) for item in sentence.split(' ')])

    # For all classes except the largest ones; add duplicate (possibly augmented) samples until all classes have the same size
    def _oversample_split(self, X_train, y_train, synonym_extra_samples = False, augment_extra_samples = False):
        """Split the oversampled train dataset

        @Param X_train: The full array of sentences

        @Param y_train: The train labels in the train dataset

        Returns: Oversampled training dataset
        """
        classes = {}
        for X, y in zip(X_train, y_train):
            if y not in classes:
                classes[y] = []
            classes[y].append(X)

        max_class_size = max([len(entries) for entries in classes.values()])

        Xs, ys = [],[] 
        for y in classes.keys():
            for i in range(max_class_size):
                sentence = classes[y][i % len(classes[y])]
                if i >= len(classes[y]):
                    if self.synonym_extra_samples:
                        sentence = self._get_synonym_sentence(sentence)
                    if self.augment_extra_samples:
                        sentence = self._get_augment_sentence(sentence)
                Xs.append(sentence)
                ys.append(y)

        #with open(filename_train+"augment", 'w', encoding='utf8') as csvFile:
        #    fileWriter = csv.writer(csvFile, delimiter='\t')
        #    for i in range(0, len(Xs)-1):
        #        fileWriter.writerow([Xs[i] + '\t' + ys[i]])

        return Xs, ys

    def _synonym_split(self, X_train, y_train, num_samples=100):
        """Split the augmented train dataset

        @Param X_train: The full array of sentences

        @Param y_train: The train labels in the train dataset

        @Param num_samples: the number of new sentences to create (default 1000)

        Returns: Augmented training dataset
        """
        Xs, ys = [], []
        for X, y in zip(X_train, y_train):
            sample = [[Xs.append(self._get_synonym_sentence(X)), ys.append(y)] for item in range(self.additional_synonyms)]
            #print(X, y)

        #with open(filename_train+"augment", 'w', encoding='utf8') as csvFile:
        #    fileWriter = csv.writer(csvFile, delimiter='\t')
        #    for i in range(0, len(Xs)-1):
        #        fileWriter.writerow([Xs[i] + '\t' + ys[i]])
        return Xs, ys

    def load(self):
        """Load the file for now only the test.csv, train.csv files hardcoded

        Returns: The vector separated in test, train and the labels for each one
        """
        with open(self.dataset_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter='	')
            all_rows = list(readCSV)
            #for i in all_rows:
                #if i ==  28823:
                    #print(all_rows[i])
            X_test = [a[0] for a in all_rows]
            y_test = [a[1] for a in all_rows]

        with open(self.dataset_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            all_rows = list(readCSV)
            X_train = [a[0] for a in all_rows]
            y_train = [a[1] for a in all_rows]
        return X_test, y_test, X_train, y_train

    def process_sentence(self, x):
        """Clean the tokens from stop words in a sentence.

        @Param x: Sentence to get rid of stop words.

        Returns: clean string sentence
        """
        clean_tokens = []
        doc = nlp.tokenizer(x)
        for token in doc:
            if not token.is_stop:
                clean_tokens.append(token.lemma_)
        return " ".join(clean_tokens)

    @staticmethod
    def get_synonyms(word, number= 3):
        """Get synonyms for a given word using synsets

        Returns: List of synonyms of length 'number'
        """
        synonyms = []
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonyms.append(l.name().lower().replace("_", " "))
        synonyms = list(OrderedDict.fromkeys(synonyms))
        return synonyms[:number]
        #return [token.text for token in most_similar(nlp.vocab[word])]

    def process_batch(self, X):
        """See the progress as is coming along.

        Returns: list of clean sentences
        """
        return [self.process_sentence(a) for a in tqdm(X)]

    def stratified_split(self):
        """Split dataset into stratified test and training sets, then remove stop word from sentences

        Returns: list of dictionaries with keys train,test and values the x and y for each one                
        """
        self.X_train, self.X_test = ([self.preprocess(sentence) for sentence in self.X_train],[self.preprocess(sentence) for sentence in self.X_test])
        print(self.X_train)
        if self.oversample:
            self.X_train, self.y_train = self._oversample_split(self.X_train, self.y_train, self.synonym_extra_samples, self.augment_extra_samples)
        if self.additional_synonyms > 0:
            self.X_train, self.y_train = self._synonym_split(self.X_train, self.y_train, self.additional_synonyms)
        if self.additional_augments > 0:
            self.X_train, self.y_train = self._augment_split(self.X_train, self.y_train, self.additional_augments)

        splits = [{"train": {"X": self.X_train, "y": self.y_train},
                "test": {"X": self.X_test, "y": self.y_test}}]
        return splits

    def get_splits(self):
        """ Get the splitted sentences
        
        Returns: splitted list of dictionaries
        """
        return self.splits

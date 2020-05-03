import spacy
class CreateData():

    def __init__(self,  vec):

        nlp=spacy.load('en_core_web_lg')
        print('Running')                     # Spacy Library token
        self.VECTORIZER = vec


    def data_for_training():
        vectorizer, feature_names = get_vectorizer(X_train_raw, preprocessor=preprocess, tokenizer=tokenize)

        X_train = vectorizer.transform(X_train_raw).toarray()
        X_test = vectorizer.transform(X_test_raw).toarray()

        return X_train, y_train_raw, X_test, y_test_raw, feature_names

    def tokenize(doc):
        """
        Returns a list of strings containing each token in `sentence`
        """
        #return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
        #                            doc) if i != '' and i != ' ' and i != '\n']
        tokens = []
        doc = nlp.tokenizer(doc)
        for token in doc:
            tokens.append(token.text)
        return tokens



    ## Comment to Explain Function
    def preprocess(doc):
        clean_tokens = []
        doc = nlp(doc)
        for token in doc:
            if not token.is_stop:
                clean_tokens.append(token.lemma_)
        return " ".join(clean_tokens)


    def get_vectorizer(corpus, preprocessor=None, tokenizer=None):
        if self.VECTORIZER == "count":
            vectorizer = CountVectorizer(analyzer='word')#,ngram_range=(1,1))
            vectorizer.fit(corpus)
            feature_names = vectorizer.get_feature_names()
        elif self.VECTORIZER == "hash":
            vectorizer = HashingVectorizer(analyzer='word', n_features=2**10, non_negative=True)
            vectorizer.fit(corpus)
            feature_names = None
        elif self.VECTORIZER == "tfidf":
            vectorizer = TfidfVectorizer(analyzer='word')
            vectorizer.fit(corpus)
            feature_names = vectorizer.get_feature_names()
        else:
            raise Exception("{} is not a recognized Vectorizer".format(VECTORIZER))
        return vectorizer, feature_names

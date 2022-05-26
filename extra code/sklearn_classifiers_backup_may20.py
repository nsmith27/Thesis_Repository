import os
import re
from posixpath import split
import scipy
import numpy as np
import pandas as pd
import pandas as pd, numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.neural_network import MLPClassifier

import text2emotion as te



class TextID:
    def __init__(self, X, y, outfile):
        self.methods = { 'Multinomial Naïve Bayes': MultinomialNB(),
                        # 'Guassian Naïve Bayes': GaussianNB(),
                        # 'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
                        'Radius Neighbor': RadiusNeighborsClassifier(radius=10.0),
                        'KNN': KNeighborsClassifier(n_neighbors=5),
                        'SVM': SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=50, tol=1e-3),
                        'Linear SVC': LinearSVC(),
                        'Random Forest Classifier': RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
                        # 'Random Forest Regressor': RandomForestRegressor(random_state=42),
                        'Logistic Regression': LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto"),
                        'MLP Classifier': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 8), random_state=1)
                        } 

        self.X = X
        self.y = y
        self.X_train = list()
        self.y_train = list()
        self.X_test = list()
        self.y_test = list()
        self.training_data = dict()
        self.report_path = str()

        self.test_train_split()
        self.setup_prediction_store(outfile)
        self.get_sentiment()
        self.get_emotion()
        self.get_CV()
        self.NLTK_train()
        return

    def setup_prediction_store(self, out):
        name = 'Actual_Values'
        y = pd.DataFrame(self.y_test, columns=[name])
        self.report_path = out + '.csv'
        y.to_csv(self.report_path)
        return 

    def save_prediction_vector(self, predicted, name):
        p = pd.DataFrame(predicted, columns=[name])
        df = pd.read_csv(self.report_path, index_col=0)
        ndf = pd.concat([df, p], axis=1 )
        ndf.to_csv(self.report_path)
        return

    def test_train_split(self):
        self.X_train, self.X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state=42)
        self.y_train = list(y_train)
        self.y_test = list(y_test)
        return

    def NLTK_train(self):
        print('Training... ')
        L = ['Sentiment', 'Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
        DFrames = {'E': ('ES_', []),
                   'S': ('S_', L[1:]),
                   'V': ('', [L[0]])
                  }
        train = self.get_dfs(self.X_train)
        test = self.get_dfs(self.X_test)
        for meta in DFrames:
            train = self.get_dfs(train, DFrames[meta][1])
            test = self.get_dfs(test, DFrames[meta][1])
            for M in self.methods:
                self.TRAIN_TEST(M, train, test, DFrames[meta][0])

        return 
    
    def get_dfs(self, source, exclusion=[]):
        # print(type(source))
        df = source.copy(deep=True)
        for i in exclusion:
            df = df.drop(i, axis=1)
        return df

    def TRAIN_TEST(self, M, X_train, X_test, name):
        X_train = scipy.sparse.csr_matrix(X_train)
        X_test = scipy.sparse.csr_matrix(X_test)
        if M in ['Guassian Naïve Bayes', 'Quadratic Discriminant Analysis']:
            self.methods[M].fit(X_train, self.y_train)
        else:
            self.methods[M].fit(X_train, self.y_train)
        predicted = self.methods[M].predict(X_test)
        if M in ['Random Forest Regressor']:
            predicted = [int(i) for i in predicted]
        name = name + M
        self.save_prediction_vector(predicted, name)
        print(name, '\n', np.mean(predicted == self.y_test), '\n')
        # report = metrics.classification_report(self.y_test, predicted)
        # print(report)
        return 

    def get_sentiment(self):
        analyzer = SentimentIntensityAnalyzer()
        train = list()
        test = list()
        for text in self.X_train['review_text']:
            sentiment = analyzer.polarity_scores(text)['compound']
            sentiment = (sentiment + 1)/2
            train.append(sentiment)
        for text in self.X_test['review_text']:
            sentiment = analyzer.polarity_scores(text)['compound']
            sentiment = (sentiment + 1)/2
            test.append(sentiment)
        self.X_train['Sentiment'] = train
        self.X_test['Sentiment'] = test
        return 

    def get_emotion(self):
        emotions = {'Angry': [], 'Fear': [], 'Happy': [], 'Sad': [], 'Surprise': [] }
        for text in self.X_train['review_text']:
            x = te.get_emotion(text)
            for key in emotions:
                emotions[key].append(x[key])
        for key in emotions:
            self.X_train[key] = emotions[key]        

        emotions = {'Angry': [], 'Fear': [], 'Happy': [], 'Sad': [], 'Surprise': [] }
        for text in self.X_test['review_text']:
            x = te.get_emotion(text)
            for key in emotions:
                emotions[key].append(x[key])
        for key in emotions:
            self.X_test[key] = emotions[key]
        
        self.save_train_test()
        return 

    def get_CV(self, tfidf=True, col='review_text'):
        Train = list(self.X_train[col])
        Test = list(self.X_test[col])
        self.X_train.drop(col, axis=1, inplace=True)
        self.X_test.drop(col, axis=1, inplace=True)
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)

        CV = CountVectorizer(ngram_range=(1,2))
        Train = CV.fit_transform(Train)
        Test = CV.transform(Test)
        TFIDF = TfidfTransformer()
        Train = TFIDF.fit_transform(Train)
        Test = TFIDF.transform(Test)

        Train = pd.DataFrame(Train.todense(), columns=CV.get_feature_names())
        Test = pd.DataFrame(Test.todense(), columns=CV.get_feature_names())
        self.X_train = pd.concat([self.X_train, Train], axis=1)
        self.X_test = pd.concat([self.X_test, Test], axis=1)
        # self.save_train_test()
        return 

    def save_train_test(self):
        train = pd.DataFrame(self.X_train)
        test = pd.DataFrame(self.X_test)
        train.to_csv('zztrain.csv')
        test.to_csv('zztest.csv')

        return


        
# Train = pd.DataFrame(Train.todense(), columns=CV.get_feature_names())
# Test = pd.DataFrame(Test.todense(), columns=CV.get_feature_names())
# self.X_train = scipy.sparse.csr_matrix(pd.DataFrame.sparse.from_spmatrix(Train))
# self.X_test = scipy.sparse.csr_matrix(pd.DataFrame.sparse.from_spmatrix(Test))


# def text_vectorizing_process(df):
#     func = lambda x: [word for word in x.split()]
#     # Using the CountVectorizer class to get a count of words from the review text
#     # Ngram_range is set to 1,2 - meaning either single or two word combination will be extracted
#     cvec = CountVectorizer(func, ngram_range=(1,2), min_df=.005, max_df=.9)
#     cvec.fit(df['review_text'])
#     # Get the total n-gram count
#     len(cvec.vocabulary_)

#     ### Term counts for each review
#     # Creating the bag-of-words representation 
#     cvec_counts = cvec.transform(data2['Review Text'])
#     print('sparse matrix shape:', cvec_counts.shape)
#     print('nonzero count:', cvec_counts.nnz)
#     print('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))
    
#     ### Calculate the weights for each term in each review
#     # Instantiating the TfidfTransformer
#     transformer = TfidfTransformer()

#     # Fitting and transforming n-grams
#     transformed_weights = transformer.fit_transform(cvec_counts)
#     transformed_weights

#     ### Get TF-IDF Matrix
#     # Getting a list of all n-grams
#     transformed_weights = transformed_weights.toarray()
#     vocab = cvec.get_feature_names()

#     # Putting weighted n-grams into a DataFrame and computing some summary statistics
#     model = pd.DataFrame(transformed_weights, columns=vocab)
#     model['Keyword'] = model.idxmax(axis=1)
#     model['Max'] = model.max(axis=1)
#     model['Sum'] = model.drop('Max', axis=1).sum(axis=1)
#     model.head(10)

#     return 

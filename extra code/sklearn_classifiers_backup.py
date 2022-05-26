import os
import re
from posixpath import split
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

from sklearn.neural_network import MLPClassifier



class TextID:
    class Method:
        def __init__(self, name, type):    
            self.name = name
            self.type = type
            self.model = None

        def set_model(self, model):
            self.model = model
            return
            
        def nltk_pipe(self):
            return (self.name, self.type)

    def __init__(self, df, outfile):
        self.methods = [ self.Method('Multinomial Naïve Bayes', MultinomialNB()),
                        # self.Method('Guassian Naïve Bayes', GaussianNB()),
                        # self.Method('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),
                        self.Method('Radius Neighbor', RadiusNeighborsClassifier(radius=10.0)),
                        self.Method('KNN', KNeighborsClassifier(n_neighbors=5)),
                        self.Method('SVM', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=50, tol=1e-3)),
                        self.Method('Linear SVC', LinearSVC()),
                        self.Method('Random Forest Classifier', RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)),
                        # self.Method('Random Forest Regressor', RandomForestRegressor(random_state=42)),
                        self.Method('Logistic Regression', LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto")),
                        self.Method('MLP Classifier', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 8), random_state=1))
                        ] 
        self.classes = set()
        self.X_train = list()
        self.y_train = list()
        self.X_test = list()
        self.y_test = list()
        self.training_data = dict()
        self.report_path = str()
        self.test_train_split(df)
        self.setup_prediction_store(outfile)
        # self.create_features()
        self.NLTK_train()
        return

    def setup_prediction_store(self, out):
        name = 'Actual_Values'
        y = pd.DataFrame(self.y_test, columns=[name])
        self.report_path = out + '.csv'
        y.to_csv(self.report_path)
        return 

    def save_prediction_vector(self, predicted, name):
        p = pd.DataFrame(predicted, columns =[name])
        df = pd.read_csv(self.report_path, index_col=0)
        ndf = pd.concat([df, p], axis=1 )
        ndf.to_csv(self.report_path)
        return

    def test_train_split(self, df):
        X = df['review_text']
        y = df['rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        self.X_train = list(X_train)
        self.X_test = list(X_test)
        self.y_train = list(y_train)
        self.y_test = list(y_test)
        return

    def NLTK_train(self):
        print('Training... ')
        for M in self.methods:
            text_clf = Pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                M.nltk_pipe()
                                ])
            M.set_model(text_clf)
            if M.name in ['Guassian Naïve Bayes', 'Quadratic Discriminant Analysis']:
                x_train = self.X_train.todense()
                M.model.fit(x_train, self.y_train)
            else:
                M.model.fit(self.X_train, self.y_train)
            predicted = M.model.predict(self.X_test)
            if M.name in ['Random Forest Regressor']:
                predicted = [int(i) for i in predicted]
            self.save_prediction_vector(predicted, M.name)
            print(M.name, '\n', np.mean(predicted == self.y_test), '\n')

        return 


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

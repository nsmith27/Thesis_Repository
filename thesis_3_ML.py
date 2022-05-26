# import os
# import re
# import nltk
# import time
# import math
# from datetime import date
# import pandas as pd

# import spacy
# import scipy
# import random
# import string
# import functools
# import numpy as np
# import seaborn as sns
# from nntplib import NNTP
# from sklearn import tree
# # nltk.download('punkt')
# import text2emotion as te
# from pydoc import describe
# # nltk.download('wordnet')
# from posixpath import split
# from sklearn import metrics
# from sklearn.svm import SVC
# # nltk.download('stopwords')
# from string import punctuation
# from sysconfig import get_path
# import matplotlib.pyplot as plt
# from autocorrect import Speller
# import pandas as pd, numpy as np
# from nltk.corpus import stopwords
# from sklearn.svm import LinearSVC
# from typing_extensions import Self
# from sklearn.decomposition import PCA
# import matplotlib.patches as mpatches
# from sklearn.pipeline import Pipeline
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from distutils.command.clean import clean
# from sklearn.feature_selection import RFE
# from wordcloud import WordCloud, STOPWORDS
# from sklearn.metrics import accuracy_score
# from sklearn.naive_bayes import GaussianNB
# pd.options.mode.chained_assignment = None
# from sklearn.feature_extraction import text
# # nltk.download('averaged_perceptron_tagger')
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import SGDClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.neighbors import RadiusNeighborsClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from factor_analyzer import FactorAnalyzer  # pip install factor_analyzer
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import re
import time
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
import multiprocessing as mp


#############################################################################################################################################
## Helper Functions                                                                                                                         #
#############################################################################################################################################
def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()

        print(f'\tFunction {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def select_columns(df, selected: list):
    if not selected:
        return df
    L_DF = []
    for s in selected:
        if type(s) is int:
            L_DF.append(df.iloc[:, [s] ])
        elif type(s) is str:
            L_DF.append(df.loc[:, [s] ])
        elif type(s) is tuple:
            A = s[0]
            B = s[1]
            if type(A) is str:
                A = df.columns.get_loc(A)
            if type(B) is str:
                B = df.columns.get_loc(B)
            if A == 0 and B == 0:
                span = df
            elif A == 0:
                span = df.iloc[:, : B ] 
            elif B == 0:
                span = df.iloc[:, A : ] 
            else:
                span = df.iloc[:, A : B ] 
            L_DF.append(span)
        elif type(s) == slice:
            span = df.iloc[:, s ]
            L_DF.append(span)
        else:
            return df
    ndf = pd.concat(L_DF, axis=1 )
    return ndf

def select_rows_by_rating(df, num_selected, selected=[1,2,3,4,5]):
    # color_or_shape = df.loc[(df['Color'] == 'Green') | (df['Shape'] == 'Rectangle')]
    # select_price = df.loc[df['Price'] >= 10]
    L_DF = []
    for i in selected:
        temp = df.loc[df['rating'] == i]
        temp = temp.sample(n= num_selected)
        L_DF.append(temp)
    ndf = pd.concat(L_DF, axis=0 )
    ndf.reset_index(inplace=True)
    return ndf 

#############################################################################################################################################
## Sentiment                                                                                                                                #
#############################################################################################################################################




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
        self.report_path = str()

        self.wrap_split_train_test()
        self.setup_prediction_store(outfile)
        self.get_sentiment('review_text', 'sentiment')
        self.get_sentiment('clean_review_text', 'clean_sentiment')
        self.get_emotion('review_text')
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

    def wrap_split_train_test(self):
        self.X_train, self.X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state=42)
        self.y_train = list(y_train)
        self.y_test = list(y_test)
        del self.X
        del self.y
        return

    def NLTK_train(self):
        # print('Training... ')
        L = ['sentiment', 'clean_sentiment', 'Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
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
        # print(name, '\n', np.mean(predicted == self.y_test), '\n')
        # report = metrics.classification_report(self.y_test, predicted)
        # print(report)
        return 

    @timer_func
    def get_sentiment(self, read_col, write_col):
        analyzer = SentimentIntensityAnalyzer()
        train = list()
        test = list()
        for text in self.X_train[read_col]:
            sentiment = analyzer.polarity_scores(text)['compound']
            sentiment = (sentiment + 1)/2
            train.append(sentiment)
        for text in self.X_test[read_col]:
            sentiment = analyzer.polarity_scores(text)['compound']
            sentiment = (sentiment + 1)/2
            test.append(sentiment)
        self.X_train[write_col] = train
        self.X_test[write_col] = test
        return 

    @timer_func
    def get_emotion(self, read_col):
        emotions = {'Angry': [], 'Fear': [], 'Happy': [], 'Sad': [], 'Surprise': [] }
        for text in self.X_train[read_col]:
            x = te.get_emotion(text)
            for key in emotions:
                emotions[key].append(x[key])
        for key in emotions:
            self.X_train[key] = emotions[key]        

        emotions = {'Angry': [], 'Fear': [], 'Happy': [], 'Sad': [], 'Surprise': [] }
        for text in self.X_test[read_col]:
            x = te.get_emotion(text)
            for key in emotions:
                emotions[key].append(x[key])
        for key in emotions:
            self.X_test[key] = emotions[key]
        
        self.save_train_test()
        return 

    def get_CV(self, tfidf=True, cols=['review_text', 'clean_review_text']):
        Train = list(self.X_train[cols[0]])
        Test = list(self.X_test[cols[0]])
        for k in cols:
            self.X_train.drop(k, axis=1, inplace=True)
            self.X_test.drop(k, axis=1, inplace=True)
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
        self.save_train_test()
        return 

    def save_train_test(self):
        train = pd.DataFrame(self.X_train)
        test = pd.DataFrame(self.X_test)
        train.to_csv('zztrain.csv')
        test.to_csv('zztest.csv')

        return





# read preprocessed data 
    # path = r'450K_prep.csv'
# get N reviews of each rating
# select nltk or spacy features 
# perform binning as specified by input (2a, 2b, 3a, 3b, 5)
# get count vectorization
# drop unnecessary columns  
# train/test split
# run all ML algorithms 




# from sklearn.svm import LinearSVC
# from typing_extensions import Self
# from sklearn.decomposition import PCA
# import matplotlib.patches as mpatches
# from sklearn.pipeline import Pipeline
# from nltk.stem import WordNetLemmatizer
# from distutils.command.clean import clean
# from sklearn.feature_selection import RFE
# from wordcloud import WordCloud, STOPWORDS
# from sklearn.metrics import accuracy_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn.feature_extraction import text
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import SGDClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.neighbors import RadiusNeighborsClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from factor_analyzer import FactorAnalyzer  # pip install factor_analyzer
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# import scipy
# import random
# import string
# import functools
# import numpy as np
# import seaborn as sns
# from nntplib import NNTP
# from sklearn import tree

# from sklearn import metrics
# from sklearn.svm import SVC

# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from pydoc import describe
# from posixpath import split
# from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
# pd.options.mode.chained_assignment = None
# import matplotlib.pyplot as plt
# from sysconfig import get_path
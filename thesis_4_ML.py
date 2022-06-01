#############################################################################################################################################
## Tasks                                                                                                                                     #
#############################################################################################################################################
# get csv data from ./output_3/
# perform binning
# run all sklearn ML algorithms
# run BERT
# for each ML algorithm, save prediction scores in file named after the ML algorithm
# save files to ./output_4/ 


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import re
import time
from datetime import datetime, date
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
## Logging Functions                                                                                                                        #
#############################################################################################################################################
def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        print_log(f'\nFunction {func.__name__!r} running...', '\t')
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        convert = time.strftime("%H:%M:%S", time.gmtime(t2-t1))
        print_log(f'Function {func.__name__!r} executed in {convert}')
        return result
    return wrap_func

print_log_path = './output_4/log.txt'
def create_log(start_time, source_file, dest_file):
    with open(print_log_path, 'a+') as file:
        border = '\n' + '='*160
        log_date = date.today().strftime("%B %d, %Y")
        log_date = border + ('\n' + log_date + ' at ' + start_time) 
        source = '\tSource: ' + source_file
        dest = '\tDestination: ' + dest_file + border
        file.write(log_date + source + dest)
    return
def print_log(S, ending=''):
    print(S, end=ending)
    S = S + ending
    with open(print_log_path, 'a+') as file:
        file.write(S)
    return 

#############################################################################################################################################
## Class                                                                                                                                    #
#############################################################################################################################################
class ML():
    def __init__(self, dir_source=r'./output_3/', size='500'):
        self.start_time = datetime.now().strftime("%H:%M")
        self.dir_source = dir_source
        self.dir_dest = r'./output_4/'
        self.size = size
        num_cores = mp.cpu_count()
        # self.methods = { 'Multinomial Naïve Bayes': MultinomialNB(n_jobs=num_cores),
        #                 # 'Guassian Naïve Bayes': GaussianNB(),
        #                 # 'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
        #                 'Radius Neighbor': RadiusNeighborsClassifier(radius=10.0, n_jobs=num_cores),
        #                 'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=num_cores),
        #                 'SVM': SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=50, tol=1e-3, n_jobs=num_cores),
        #                 'Linear SVC': LinearSVC(n_jobs=num_cores),
        #                 'Random Forest Classifier': RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0, n_jobs=num_cores),
        #                 # 'Random Forest Regressor': RandomForestRegressor(random_state=42),
        #                 'Logistic Regression': LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto", n_jobs=num_cores),
        #                 'MLP Classifier': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 8), random_state=1, n_jobs=num_cores)
        #                 } 
        self.all_current_train_columns = None
        self.all_current_test_columns = None
        self.X_train = list()
        self.y_train = list()
        self.X_test = list()
        self.y_test = list()
        self.report_path = str()

        create_log(self.start_time, self.dir_source, self.dir_dest)
        
        self.bins = [{'rating': {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}},
                     {'rating': {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}},
                     {'rating': {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}},
                     {'rating': {1: 0, 2: 1, 3: 1, 4: 1, 5: 2}},
                     None]
        self.drop_columns = {'Emotion': ['rating', 'clean_nltk', 'clean_spacy'],
                             'Sentiment' : ['rating', 'clean_nltk', 'clean_spacy', 'Happy', 'Angry', 'Surprise', 'Sad', 'Fear'], 
                             'Basic' : ['rating', 'clean_nltk', 'clean_spacy', 'Happy', 'Angry', 'Surprise', 'Sad', 'Fear', 'sentiment_nltk', 'sentiment_spacy']}
        for lib in ['nltk', 'spacy']:
            for bIndex, bVal in enumerate(self.bins):
                for d in self.drop_columns:
                    self.get_df(lib, bVal, self.drop_columns[d])
                    self.save_Xy(lib, bIndex, d)
                    
                    # for m in self.methods:
                    #     self.setup_prediction_store(lib, bIndex)
                    #     pass
        return

    def get_df(self, lib, bins, drop_columns):
        # There are two paths from the following set--one train and one test
        T = [i for i in os.listdir(self.dir_source) if lib in i and self.size in i]
        if len([i for i in T if 'test' in i or 'train' in i]) > 2:
            print('Ambiguous input.  Program exiting...')
            exit()
        for t in T:
            if 'train' in t:
                self.X_train = pd.read_csv(self.dir_source + t, index_col=[0])
                self.X_train = self.bin_df(self.X_train, bins)
                self.y_train = self.X_train['rating'].to_numpy()
                self.X_train = self.X_train.drop(drop_columns, axis=1, errors='ignore')
                self.all_current_train_columns = list(self.X_train.columns.insert(0, 'rating'))
                self.X_train = self.X_train.to_numpy()
            elif 'test' in t:
                self.X_test = pd.read_csv(self.dir_source + t, index_col=[0])
                self.X_test = self.bin_df(self.X_test, bins)
                self.y_test = self.X_test['rating'].to_numpy()
                self.X_test = self.X_test.drop(drop_columns, axis=1, errors='ignore')
                self.all_current_test_columns = list(self.X_test.columns.insert(0, 'rating'))
                self.X_test = self.X_test.to_numpy()        
        return 

    def bin_df(self, df, bins):
        if bins:
            for key in bins:
                df[key] = df[key].map(bins[key])
        return df

    @timer_func
    def save_Xy(self, lib, bin, drop_columns):
        B = {0: '2A', 1: '2B', 2: '3A', 3: '3B', 4: '5'}
        name = 'intermediate/' + lib + '_' + B[bin] + '_' + drop_columns
        all_data = np.insert(self.X_train, 0, self.y_train, axis=1)
        all_data = pd.DataFrame(all_data, columns = self.all_current_train_columns)
        all_data.to_csv(self.dir_dest + name + '_X_train.csv', index=False)

        all_data = np.insert(self.X_test, 0, self.y_test, axis=1)
        all_data = pd.DataFrame(all_data, columns = self.all_current_test_columns)
        all_data.to_csv(self.dir_dest + name + '_X_test.csv', index=False)

        return

######################################################################################################################################################
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

#############################################################################################################################################
## Main                                                                                                                                     #
#############################################################################################################################################
if __name__ == '__main__':
    mp.freeze_support()
    ML(size='500')
    # ML(size='5K')
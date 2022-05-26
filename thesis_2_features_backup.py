def warn(*args, **kwargs):
    pass
from distutils.command.build_scripts import first_line_re
import warnings
warnings.warn = warn

import os
import platform
import re
import nltk
import time
import math
from datetime import date
import pandas as pd
import shutil

import spacy
import scipy
import random
import string
import functools
import numpy as np
import seaborn as sns
from nntplib import NNTP
from sklearn import tree
# nltk.download('punkt')
import text2emotion as te
from pydoc import describe
# nltk.download('wordnet')
from posixpath import split
from sklearn import metrics
from sklearn.svm import SVC
# nltk.download('stopwords')
from string import punctuation
from sysconfig import get_path
import matplotlib.pyplot as plt
from autocorrect import Speller
import pandas as pd, numpy as np
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from typing_extensions import Self
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from distutils.command.clean import clean
from sklearn.feature_selection import RFE
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
pd.options.mode.chained_assignment = None
from sklearn.feature_extraction import text
# nltk.download('averaged_perceptron_tagger')
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from factor_analyzer import FactorAnalyzer  # pip install factor_analyzer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

import multiprocessing as mp


class Preprocess():
    # Before -- DF.columns
    #   {rating (int), review_text (str)} 
    # After  -- DF.columns
    #   {rating (int), review_text (str), clean_nltk (str), clean_spacy (str), 
    #    sentiment (float), sentiment_nltk (float), sentiment_spacy (float), 
    #    angry (float), fear (float), happy (float), sad (float), surpise (float)}
    
    def __init__(self,path_source, path_dest):
        self.t0_class = time.time()
        self.path_source = path_source
        self.path_dest = path_dest
        self.df = None

        self.get_df()
        self.wrap_clean('clean_nltk')
        self.wrap_clean('clean_spacy', 10**3)
        return 
        get_sentiment(df, '')
        get_sentiment(df, 'nltk')
        get_sentiment(df, 'spacy')    

        save_df(df)


        return

    #############################################################################################################################################
    ## Helper Functions                                                                                                                         #
    #############################################################################################################################################
    def timer_func(func):
        # This function shows the execution time of 
        # the function object passed
        def wrap_func(*args, **kwargs):
            print(f'Function {func.__name__!r} running...', end='\t')
            t1 = time.time()
            result = func(*args, **kwargs)
            t2 = time.time()
            convert = time.strftime("%M:%S", time.gmtime(t2-t1))
            print(f'executed in {convert} (min:sec)')
            return result
        return wrap_func

    def get_df(self):
        if not os.path.exists(self.path_source):
            self.df = pd.read_csv(r'450K_reviews.csv', index_col=0)
        elif os.path.exists(self.path_dest):
            today = date.today()
            today = today.strftime("%b_%d_")
            shutil.copyfile(self.path_dest, today + self.path_dest)
            self.df = pd.read_csv(self.path_source, index_col=0)
        else:
            self.df = pd.read_csv(self.path_source, index_col=0)
        self.df.fillna('', inplace=True)
        return 

    def find_left_off(self, col):
        kept = []
        for i, v in enumerate(self.df[col]):
            if len(v):  
                kept.append(v)
            else:
                break
        return i, kept

    def save_df(self, path, time=False):
        slash = '/'
        if 'Windows' in platform.platform() :
            slash = '\\'
        location = os.getcwd() + slash + path
        print('...saving dataframe at ' + location)
        self.df.to_csv(path)
        if time:
            t2 = time.time()
            convert = time.strftime("%M:%S", time.gmtime(t2-self.t0_class))
            print(f'Time since Preprocessing began: {convert} (min:sec)')
        return

    ############################################################################################################################################
    ## Clean Text                                                                                                                              #
    # Normalizing case                                                                                                                         #
    # Remove extra line breaks                                                                                                                 #
    # Tokenize                                                                                                                                 #
    # Remove stop words and punctuations                                                                                                       #
    ############################################################################################################################################
    @timer_func
    def wrap_clean(self, new_col, chunk_size=450_000/12):
        func = None
        if new_col == 'clean_nltk':
            func = self.clean_nltk
        elif new_col == 'clean_spacy':
            func = self.clean_spacy

        if new_col not in self.df:
            self.df[new_col] = ''
        num_cpu = mp.cpu_count()
        size = len(self.df.index)
        chunk_size = math.ceil(chunk_size)
        save_size = chunk_size * num_cpu
        x = list(self.df['review_text'])
        while True:
            first_miss, group_begin = self.find_left_off(new_col)
            if first_miss >= size-1:
                break
            depth = min(first_miss + save_size, size)
            input = [x[a:a+chunk_size] for a in range(first_miss, depth, chunk_size)]
            group_end = list(self.df[new_col])[depth:size]
            # Parallization
            with mp.Pool(num_cpu) as pool:
                result = pool.map(func, input)
                result.insert(0, group_begin)
                result.insert(depth, group_end)
                self.df[new_col] = [item for sublist in result for item in sublist]
            self.save_df(self.path_dest)
        return 

    def clean_nltk(self, L):
        t1 = time.time()
        new_col = 'clean_nltk'
        out = []
        for i in range(len(L)):
            text = L[i]
            text = self.replace_web_reference(text)
            text = self.tokenize(text)
            text = self.normalize_case(text) 
            # text = spell_correct(text)
            text = self.remove_punc(text)
            text = self.remove_stop(text)
            text = ' '.join(text) if len(text) > 0 else L[i]
            out.append(text)
        t2 = time.time()
        print(f'\tFunction clean_nltk executed in {(t2-t1):.4f}s')
        return out
    
    def clean_spacy(self, L):
        t1 = time.time()
        new_col = 'clean_spacy'
        out = []
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        for i in range(len(L)):
            text = L[i]
            text = self.replace_web_reference(text)
            text = nlp(text)
            text = [word.lemma_ for word in text]
            text = self.normalize_case(text)
            text = self.remove_punc(text)
            text = self.remove_stop(text)
            text = ' '.join(text) if len(text) > 0 else L[i]
            out.append(text)
        t2 = time.time()
        print(f'\tFunction clean_nltk executed in {(t2-t1):.4f}s')
        return out

    def replace_web_reference(self, text):
        replacement = '-ref_website-'
        pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        try:
            text = re.sub(pattern, replacement, text)
        except:
            print('exception>>>>>>>>>>>')
        return text

    def tokenize(self, text):
        return word_tokenize(text)

    def normalize_case(self, text):
        return [t.lower() for t in text]

    def remove_punc(self, text):
        punct = list(punctuation) + ["...", "``","''", "===="]
        return [i for i in text if str(i) not in punct]

    def spell_correct(self, text):
        spell = Speller(lang='en')
        for i, v in enumerate(text):
            if v == 'ref_website':
                continue
            text[i] = spell(v)

        return text

    def remove_stop(self, text):
        stop_words = stopwords.words("english")
        keep = '''
        no
        not
        don
        dont
        don't
        won
        wont
        won't
        shouldn
        shouldnt
        shouldn't
        too
        '''.split('\n')[1:-1]
        for i in keep:
            if i in stop_words:
                stop_words.remove(i)

        return [i for i in text if i not in stop_words]


    #############################################################################################################################################
    ## Sentiment and Emotion                                                                                                                    #
    #############################################################################################################################################
    # def get_sentiment(self, read_col, write_col):
    #     if write_col not in df:
    #         df[write_col] = None
    #     analyzer = SentimentIntensityAnalyzer()
    #     for index, text in df[read_col].iterrows():
    #         if df.at[index, write_col]:
    #             print('HI!!')
    #             continue
    #         score = analyzer.polarity_scores(text)['compound']
    #         score= (score + 1)/2
    #         df.at[index, write_col] = score

    #     return 

    # @timer_func
    # def get_emotion(self, read_col):
        # if 'clean_nltk' not in df:

    #     emotions = {'Angry': [], 'Fear': [], 'Happy': [], 'Sad': [], 'Surprise': [] }
    #     for text in self.X_train[read_col]:
    #         x = te.get_emotion(text)
    #         for key in emotions:
    #             emotions[key].append(x[key])
    #     for key in emotions:
    #         self.X_train[key] = emotions[key]        

    #     emotions = {'Angry': [], 'Fear': [], 'Happy': [], 'Sad': [], 'Surprise': [] }
    #     for text in self.X_test[read_col]:
    #         x = te.get_emotion(text)
    #         for key in emotions:
    #             emotions[key].append(x[key])
    #     for key in emotions:
    #         self.X_test[key] = emotions[key]
        
    #     self.save_train_test()
    #     return 

    # @timer_func
    # def wrap_clean_spacy(df, path):
    #     @sub_timer
    #     def clean_spacy(section):
    #         (start, end) = section   
    #         new_col = 'clean_spacy'
    #         if new_col not in df:
    #             df[new_col] = None
    #         nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    #         for i, v in enumerate(df['review_text']):
    #             if df.at[i, new_col] or i < start or i > end:
    #                 continue
    #             text = v
    #             text = replace_web_reference(text)
    #             text = nlp(text)
    #             text = [word.lemma_ for word in text]
    #             text = normalize_case(text)
    #             text = remove_punc(text)
    #             text = remove_stop(text)
    #             text = ' '.join(text)
    #             df['clean_spacy'][i] = text 
    #         return df
    #     # Parallization
    #     pool = mp.Pool(mp.cpu_count())
    #     pool.map(clean_spacy, [path for path in paths])
    #     pool.close()
    #     save_df(df, path)
    #     return df


#############################################################################################################################################
## Main                                                                                                                                     #
#############################################################################################################################################
if __name__ == '__main__':
    mp.freeze_support()
    # Preprocess(path_source=r'450K_reviews.csv', path_dest=r'450K_prep.csv')
    Preprocess(path_source=r'450K_prep.csv', path_dest=r'450K_prep.csv')

# get data from csv 
    # path_450K = r'450K_reviews.csv'
# create new column of cleaned text NLTK
# creat a new column of cleaned text SPACY
# create column of sentiment from plain text
# create column of sentiment from clean_NLTK
# create column of sentiment from clean_SPACY
# create columns of emotion from plain text
# save dataframe as 450K_prep.csv



from distutils.command.clean import clean
import re
import os
from sysconfig import get_path

import string
import random
import functools
from typing_extensions import Self
import numpy as np
import pandas as pd
import pandas as pd, numpy as np
pd.options.mode.chained_assignment = None  

import seaborn as sns
import matplotlib.patches as mpatches
from wordcloud import WordCloud, STOPWORDS
from nntplib import NNTP
from pydoc import describe
from string import punctuation
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

import spacy

from autocorrect import Speller

from sklearn import tree
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_extraction import text
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from factor_analyzer import FactorAnalyzer  # pip install factor_analyzer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn_classifiers import *
from raw_reader import *



############################################################################################################################################
## Clean Text 
# Normalizing case
# Remove extra line breaks
# Tokenize
# Remove stop words and punctuations
############################################################################################################################################
def clean_spaCy(df, path_no_extension=False, save=True):    
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for i, v in enumerate(df['review_text']):
        text = v
        text = replace_web_reference(text)
        text = nlp(text)
        text = [word.lemma_ for word in text]
        text = normalize_case(text)
        text = remove_punc(text)
        text = remove_stop(text)
        text = ' '.join(text)
        df['review_text'][i] = text 
    if path_no_extension and save:
        cwd = os.getcwd()
        df.to_csv(cwd + '/cleaned reviews/clean_' + path_no_extension + '.csv')
    return df

def clean_NLTK(df, path_no_extension=False, save=True):
    for i, v in enumerate(df['review_text']):
        text = v
        text = replace_web_reference(text)
        text = tokenize(text)
        text = normalize_case(text)
        # text = spell_correct(text)
        text = remove_punc(text)
        text = remove_stop(text)
        out = ' '.join(text)
        df['review_text'][i] = out
    if path_no_extension and save:
        cwd = os.getcwd()
        df.to_csv(cwd + '/cleaned reviews/clean_' + path_no_extension + '.csv')
    return df 

def replace_web_reference(text):
    replacement = '-ref_website-'
    pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    try:
        text = re.sub(pattern, replacement, text)
    except:
        print('exception>>>>>>>>>>>')
    return text

def tokenize(text):
    return word_tokenize(text)

def normalize_case(text):
    return [t.lower() for t in text]

def remove_punc(text):
    punct = list(punctuation) + ["...", "``","''", "===="]
    return [i for i in text if str(i) not in punct]

def spell_correct(text):
    spell = Speller(lang='en')
    for i, v in enumerate(text):
        if v == 'ref_website':
            continue
        text[i] = spell(v)

    return text

def remove_stop(text):
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
## Choose Features                                                                                                                          #
# Select 'Vanilla' Features                                                                                                                 #
# Set whether Sentiment Fatures or not                                                                                                      #
# Set Ratings for Testing { (0,1), (1,2,3), (1,2,3,4,5),... }                                                                               #
#############################################################################################################################################
def rating_binner(df, columns):
    for key in columns:
        df[key] = df[key].map(columns[key])
        df[key] = df[key].fillna(0)
    return df

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

def reduce_2Abin(df):
    ordinal = {'rating': {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}}
    return rating_binner(df, ordinal)

def reduce_2Bbin(df):
    ordinal = {'rating': {1: 0, 2: 0, 3: 1, 4: 1, 5: 1}}
    return rating_binner(df, ordinal)

def reduce_3Abin(df):
    ordinal = {'rating': {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}}
    return rating_binner(df, ordinal)

def reduce_3Bbin(df):
    ordinal = {'rating': {1: 0, 2: 1, 3: 1, 4: 1, 5: 2}}
    return rating_binner(df, ordinal)


#############################################################################################################################################
## Main                                                                                                                                     #
#############################################################################################################################################
def run(source, N, bin='5', tool='NLTK', save=True):
    path_N = str(int(N*5/1000)) + r'K_reviews.csv'
    path_no_extension = str(int(N*5/1000)) + r'K'
    df = pd.read_csv(source)
    df = select_rows_by_rating(df, N)
    # Bin review column or not
    bin = str(bin)
    special = ''
    if bin == '2a':
        df = reduce_2Abin(df)
        special = '2Abin'
    elif bin == '2b':
        df = reduce_2Bbin(df)
        special = '2Bbin'
    elif bin == '3a':
        df = reduce_3Abin(df)
        special = '3Abin'
    elif bin == '3b':
        df = reduce_3Bbin(df)
        special = '3Bbin'
    else:
        special = '5bin'

    # Choose which library/package to use
    tool = tool.lower()
    if tool == 'nltk':
        path_no_extension = path_no_extension + '_NLTK_' + special
        df = clean_NLTK(df, path_no_extension, save)
    elif tool == 'spacy':
        path_no_extension = path_no_extension + '_SPACY_' + special
        df = clean_spaCy(df, path_no_extension, save)

    X = select_columns(df, [(2,0)])
    y = df['rating']
    TextID(X, y, path_no_extension)

    return 


if __name__ == "__main__":
    path_all = r'all_reviews.csv'
    path_all_valid = r'all_valid_reviews.csv'
    path_450K = r'450K_reviews.csv'
    # convert_store_raw_data(path_450K, 90_000)

    N = 1000
    run(path_450K, N, '2a', 'NLTK')
    run(path_450K, N, '2b', 'NLTK')
    run(path_450K, N, '3a', 'NLTK')
    run(path_450K, N, '3b', 'NLTK')
    run(path_450K, N, '5', 'NLTK')

    run(path_450K, N, '2a', 'SPACY')
    run(path_450K, N, '2b', 'SPACY')
    run(path_450K, N, '3a', 'SPACY')
    run(path_450K, N, '3b', 'SPACY')
    run(path_450K, N, '5', 'SPACY')










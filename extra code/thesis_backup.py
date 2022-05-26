from distutils.command.clean import clean
import re
import os
from sysconfig import get_path

import string
import random
import functools
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
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

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



############################################################################################################################################
## Get Data 
# Raw Data reader 
# Store all in csv file
############################################################################################################################################
def get_paths(path=False, ftypes=False):
    if path:
        result = [i for i in os.path(path) if ftypes == i[-4:] ]
    else:
        cwd = os.path.dirname(os.path.realpath(__file__))
        folder = 'reviews'
        dir_path = cwd + '\\' + folder + '\\'
    paths = os.listdir(dir_path)
    L = [(dir_path + i) for i in paths if ftypes == i[-3:] ]
    return L

def read_file(path):
    result = ''
    with open(path, 'r', encoding='utf8') as file:
        result = file.read()
    return result 

def parse_review(content, delimiter=''):
    content = content.split('Review:')
    result = []
    if delimiter[-2:] != ':\n':
        delimiter += ':\n'
    for i in content:
        rating = re.findall(r'Rating:\n(\d)*', i)
        text = re.findall(r'text:\n(.*)', i)
        if rating and text:
            text = text[0].strip()
            if text:
                tup = (rating[0], '\"' + text.replace('\"', "\'") + '\"')
                result.append(tup) 
    return result

def save_to_csv(path, D):
    with open(path, 'w', encoding='utf-8') as file:
        columns =  'rating,review_text\n'
        file.write(columns)
        for n in D:
            for k in range(len(D[n])):
                input = n + ',' + D[n][k] + '\n'
                file.write(input)
    return

def convert_store_raw_data(out_path, count=0):
    paths = get_paths(ftypes='txt')
    D = {'1':[], '2':[], '3':[], '4':[], '5':[]}
    for p in paths:
        content = read_file(p)
        reviews = parse_review(content, 'Rating')
        for r in reviews:
            if len(D[r[0]]) < count or count == 0:
                D[r[0]].append(r[1])
    # for i in D:
    #     print(i, D[i][0])
    save_to_csv(out_path, D)
    return 

############################################################################################################################################
## Descriptive/Exploratory Data Analysis 
# Document term matrix using TfIdf
############################################################################################################################################


############################################################################################################################################
## Clean Text 
# Normalizing case
# Remove extra line breaks
# Tokenize
# Remove stop words and punctuations
############################################################################################################################################
def clean_spaCy(df):    
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

    df.to_csv('clean_spaCy_' + path_N)
    return df

def clean_NLTK(df):
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
    df.to_csv('clean_NLTK_' + path_N)
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
    punct = list(punctuation) + ["...", "``","''", "====", "must"]
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


############################################################################################################################################
## Choose Features
# Select 'Vanilla' Features 
# Set whether Sentiment Fatures or not
# Set Ratings for Testing { (0,1), (1,2,3), (1,2,3,4,5),... }
############################################################################################################################################


def text_vectorizing_process(df):
    func = lambda x: [word for word in x.split()]
    # Using the CountVectorizer class to get a count of words from the review text
    # Ngram_range is set to 1,2 - meaning either single or two word combination will be extracted
    cvec = CountVectorizer(func, ngram_range=(1,2), min_df=.005, max_df=.9)
    cvec.fit(df['review_text'])
    # Get the total n-gram count
    len(cvec.vocabulary_)

    ### Term counts for each review
    # Creating the bag-of-words representation 
    cvec_counts = cvec.transform(data2['Review Text'])
    print('sparse matrix shape:', cvec_counts.shape)
    print('nonzero count:', cvec_counts.nnz)
    print('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))
    
    ### Calculate the weights for each term in each review
    # Instantiating the TfidfTransformer
    transformer = TfidfTransformer()

    # Fitting and transforming n-grams
    transformed_weights = transformer.fit_transform(cvec_counts)
    transformed_weights

    ### Get TF-IDF Matrix
    # Getting a list of all n-grams
    transformed_weights = transformed_weights.toarray()
    vocab = cvec.get_feature_names()

    # Putting weighted n-grams into a DataFrame and computing some summary statistics
    model = pd.DataFrame(transformed_weights, columns=vocab)
    model['Keyword'] = model.idxmax(axis=1)
    model['Max'] = model.max(axis=1)
    model['Sum'] = model.drop('Max', axis=1).sum(axis=1)
    model.head(10)

    return 

def one_hot(df, columns):
    for i in columns:
        # df[i] = df[i].str.replace(r'unknown', 'unkn', regex=True)
        if i not in ['month', 'day_of_week']:
            df[i] = df[i].apply(lambda x: i + '_' + x)
    # Create one-hot encoding of the different categories.
    ohe = OneHotEncoder()
    feature_array = ohe.fit_transform(df[columns]).toarray()
    feature_labels = functools.reduce(lambda a,b : a+b, [list(i) for i in ohe.categories_])
    ndf = pd.DataFrame(feature_array, columns=feature_labels)
    df = pd.concat([df, ndf], axis=1)
    return df

def ordinal_encoder(df, columns):
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
            if A == 0:
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

############################################################################################################################################
## Machine Learning Models
# Separate X and Y and perform train test split, { (70-30), (80,20) }
# Parameter Tuning 
# Neural Network -- MLP, RNN, CNN
# Transformer Model
# Support Vector Machine
# Random Forest
# Logistic Regression
# KNN
# Naive Bayes
# QDA (Quadratic Discriminate Analysis)
# BERT
############################################################################################################################################


############################################################################################################################################
## Using the best estimator to make predictions on the test set
############################################################################################################################################


############################################################################################################################################
## Identifying mismatch cases
############################################################################################################################################




############################################################################################################################################
## Main
############################################################################################################################################
if __name__ == "__main__":
    N = 1_000
    path_N = str(int(N*5/1000)) + r'K_reviews.csv'
    path_all = r'all_reviews.csv'
    path_all_valid = r'all_valid_reviews.csv'
    path_450K = r'450K_reviews.csv'
    # convert_store_raw_data(path_450K, 90_000)
    df = pd.read_csv(path_450K)
    df = select_rows_by_rating(df, N)

    df = clean_NLTK(df)
    df = clean_spaCy(df)

    # print(df)
    


    df.to_csv('clean_' + path_N)
    # x = df['rating'].value_counts()









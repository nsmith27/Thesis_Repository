# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')


# # import math 
# # num_cpu = 12
# # size = 450
# # chunk_size = math.ceil(size/num_cpu)
# # x = [(i, i+chunk_size) for i in range(0, size, chunk_size)]

# # # print(x)



# #     # def sub_timer(func):
# #     #     # This function shows the execution time of 
# #     #     # the function object passed
# #     #     def wrap_func(*args, **kwargs):
# #     #         t1 = time.time()
# #     #         result = func(*args, **kwargs)
# #     #         t2 = time.time()
# #     #         print(f'\tFunction {func.__name__!r} executed in {(t2-t1):.4f}s')
# #     #         return result
# #     #     return wrap_func

# import pandas as pd
# df = pd.read_csv(r'450K_reviews.csv')
# x = list(df['review_text'])[:20]

# for i, v in enumerate(df['review_text']):
#     print(i, type(v))
#     if i > 20:
#         break

# quit()

# # size = len(x)
# # chunk = 3
# # a = [x[a:a+chunk] for a in range(0, size, chunk)]

# # for i in a:
# #     print(len(i))
# # # df = df.iloc[5:13]


# # a = [1]
# # b = [['a', 'b', 'c'],['d', 'e', 'f'],['g', 'h', 'i']]
# # b.insert(0, a)
# # print(b)
# # b = [item for sublist in b for item in sublist]
# # print(b)


# import platform

# x = platform.platform()
# print(x)
# print('Windows' in x )

# import numpy as np

# a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# c = 2
# b = [a[i:i+c] for i in range(0, len(a), c)]

# b = {'a':[1,2,3,4], 'b':[5,6,7], 'c':[8,9,10]}
# for i in b.keys():
#     print(i)

# import text2emotion as te
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# analyzer = SentimentIntensityAnalyzer()

# text1 = 'This cookies were just blehhhh.  Not the brownie-type batter I was expecting.  I followed the recipe exactly, too.  I think they might be better with margarine or butter instead of the shortening.'
# text2 = 'cookies blehhhh brownie-type batter expecting followed recipe exactly think might better margarine butter instead shortening'
# text3 = 'cookie blehhhh   brownie type batter expect   follow recipe exactly   think might well margarine butter instead shortening'

# text1 = 'Too rich, sticky and thick. Nobody liked it.'
# text2 = 'rich sticky thick nobody liked'
# text3 = 'rich sticky thick nobody like'

# t1 = analyzer.polarity_scores(text1)['compound']
# t2 = analyzer.polarity_scores(text2)['compound']
# t3 = analyzer.polarity_scores(text3)['compound']
# print(t1)
# print(t2)
# print(t3)

# e1 = te.get_emotion(text1)
# e2 = te.get_emotion(text1)
# e3 = te.get_emotion(text1)
# print(e1)
# print(e2)
# print(e3)


# a = '''
# I made this exactly as written except for reducing sugar. After two hours in the oven the outside was a hard shell and the inside was sludge. Now I am searching for a new recipe.	made exactly written except reducing sugar two hours oven outside hard shell inside sludge searching new recipe	make exactly write except reduce sugar two hour oven outside hard shell inside sludge search new recipe
# '''.replace('\n', '').split('\t')
# print(len(a))
# for i in a:
#     print((analyzer.polarity_scores(i)['compound']+1)/2, i)
# print(te.get_emotion(a[0]))

# a = 'review_text	clean_nltk	clean_spacy	sentiment_text	sentiment_nltk	sentiment_spacy	Happy	Angry	Surprise	Sad	Fear'
# a = a.split('\t')
# print(a)


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

# def f(a):
#     print('f', a)
#     return

# def g(a):
#     print('g', a)
#     return 

# x = [f, g]

# for i in x:
#     i(2)
    
import os
from re import A
import pandas as pd
import numpy as np

# size = '5'
# T = [i for i in os.listdir(r'./output_3') if 'nltk' in i and size in i]

# print(T)

df = pd.read_csv(r'./output_3/500_nltk_test.csv')
# print(df.columns)
# df = df.iloc[:, 1: ].to_numpy()
# df = df['rating'].tolist()
# print(df)

# drop_columns = {'Emotion': ['clean_nltk', 'clean_spacy'],
#                 'Sentiment' : ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear'], 
#                 'Basic' : ['sentiment_nltk', 'sentiment_spacy']}

# for i in drop_columns:
#     for k in df.columns:
#         if i in k:
#             df.drop(k, axis=1, inplace=True)
#     print(df.columns)




# df.drop(["Difficulty_Score", "Type"], axis = 1, inplace = True)
# bin = {'a': 1, 'b': 2}

# if bin:
#     print(1)
# else:
#     print(2)


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create and generate a word cloud image:
text = '''
'''
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
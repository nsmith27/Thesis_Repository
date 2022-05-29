#############################################################################################################################################
## Tasks                                                                                                                                    #
#############################################################################################################################################
# get csv data from ./output_2/
# perform bag of words with NLTK data
# perform bag of words with SPACY data
# convert input to sparse matrix
# train/test split (70/30)
# save the 4 dataframes in ./output_3/


import os
import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


#############################################################################################################################################
## Class                                                                                                                                    #
#############################################################################################################################################
class Setup():
    # Before  -- DF.columns
    #   {rating (int), review_text (str), clean_nltk (str), clean_spacy (str), 
    #    sentiment (float), sentiment_nltk (float), sentiment_spacy (float), 
    #    angry (float), fear (float), happy (float), sad (float), surpise (float)}
    # After  -- DF_NLTK.columns or DF_SPACY.columns 
    #   {rating (int), clean_nltk (str), sentiment (float), sentiment_nltk (float), 
    #    angry (float), fear (float), happy (float), sad (float), surpise (float),
    #    [bag of words columns] (0 or 1)}

    def __init__(self, path_source=r'./output_2/450K_prep.csv', dir_dest=r'./output_2/'):
        self.df_source = None
        self.get_df()
        self.get_CV('nltk')
        self.get_CV('spacy')
        return

    def get_df(self):
        return

    def get_CV(self, name):
        if 'nltk' in name:
            col = 'clean_nltk'
        elif 'spacy' in name:
            col = 'clean_spacy'

        data = list(self.df_source[col])
        count = CountVectorizer()
        word_count=count.fit_transform(text)

        TFIDF = TfidfVectorizer()
        X = TFIDF.fit_transform(data)

        Train = pd.DataFrame(data.todense(), columns=CV.get_feature_names())
        Test = pd.DataFrame(Test.todense(), columns=CV.get_feature_names())
        self.X_train = pd.concat([self.X_train, Train], axis=1)
        self.X_test = pd.concat([self.X_test, Test], axis=1)
        self.save_train_test()
        return 

    def save_df(self):

        # for k in cols:
        #     self.X_train.drop(k, axis=1, inplace=True)
        #     self.X_test.drop(k, axis=1, inplace=True)

        return


    def select_columns(self, selected: list):
        if not selected:
            return self.df_source
        L_DF = []
        for s in selected:
            if type(s) is int:
                L_DF.append(self.df_source.iloc[:, [s] ])
            elif type(s) is str:
                L_DF.append(self.df_source.loc[:, [s] ])
            elif type(s) is tuple:
                A = s[0]
                B = s[1]
                if type(A) is str:
                    A = self.df_source.columns.get_loc(A)
                if type(B) is str:
                    B = self.df_source.columns.get_loc(B)
                if A == 0 and B == 0:
                    span = self.df_source
                elif A == 0:
                    span = self.df_source.iloc[:, : B ] 
                elif B == 0:
                    span = self.df_source.iloc[:, A : ] 
                else:
                    span = self.df_source.iloc[:, A : B ] 
                L_DF.append(span)
            elif type(s) == slice:
                span = self.df_source.iloc[:, s ]
                L_DF.append(span)
            else:
                return self.df_source
        ndf = pd.concat(L_DF, axis=1 )
        return ndf


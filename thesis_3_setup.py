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
from sklearn.model_selection import train_test_split
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

    def __init__(self, path_source=r'./output_2/450K_prep.csv', dir_dest=r'./output_3/'):
        self.path_source = path_source
        self.path_dest = None
        self.df_source = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        if self.get_df():
            self.set_path_dest() 
            self.split(0.3)
            self.get_CV('nltk')
            self.get_CV('spacy')
        return

    def get_df(self):
        if not os.path.exists(self.path_source):
            print('\nPath ' + self.path_source + ' does not exist.')
            print('Exiting program...\n')
            return False
        self. df_source = pd.read_csv(self.path_source)
        return True

    def set_path_dest(self):
        full_path = self.path_source.split('/')
        file_name = full_path[-1].split('_')[0]
        self.path_dest = '/'.join(full_path[:-1]) + '/' + file_name
        return 

    def split(self, test_size=0.3):
        X = self.select_columns( [slice(1, None)] )
        y = self.df_source['rating']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        self.df_source = None
        print(type(self.y_train))
        exit()
        return 

    def get_CV(self, name):
        if 'nltk' in name:
            cols = ['clean_nltk', 'sentiment_text', 'sentiment_nltk', 'Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
        elif 'spacy' in name:
            cols = ['clean_spacy', 'sentiment_text', 'sentiment_spacy', 'Happy', 'Angry', 'Surprise', 'Sad', 'Fear']

        TFIDF=TfidfVectorizer(use_idf=True)
        Train = self.X_train[cols[0]]
        Test = self.X_test[cols[0]]
        Train = TFIDF.fit_transform(Train)
        Test = TFIDF.transform(Test)

        Train = pd.DataFrame(Train.todense(), columns=TFIDF.get_feature_names())
        Test = pd.DataFrame(Test.todense(), columns=TFIDF.get_feature_names())
        self.X_train = pd.concat([self.y_train, self.X_train, Train], axis=1)
        self.X_test = pd.concat([self.y_test, self.X_test, Test], axis=1)

        self.save_train_test()
        return 

    def save_df(self):

        # for k in cols:
        #     self.X_train.drop(k, axis=1, inplace=True)
        #     self.X_test.drop(k, axis=1, inplace=True)

        return


    def select_columns(self, df, selected: list):
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


#############################################################################################################################################
## Main                                                                                                                                     #
#############################################################################################################################################
if __name__ == '__main__':
    Setup(path_source=r'./output_2/500_prep.csv', dir_dest=r'./output_3/')
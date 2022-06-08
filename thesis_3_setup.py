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
import time
import threading
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer



#############################################################################################################################################
## Timing Function                                                                                                                          #
#############################################################################################################################################
def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

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
    #    [bag of words columns] (0 to 1)}

    def __init__(self, path_source=r'./output_2/450K_prep.csv', dir_dest=r'./output_3/'):
        t1 = time.time()
        self.path_source = path_source
        self.dir_dest = dir_dest
        self.path_dest = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        if self.get_df():
            self.set_path_dest() 
            self.split(0.3)
            self.get_CV('nltk')
            self.get_CV('spacy')
        t2 = time.time()
        print(f'Setup executed in {(t2-t1):.4f}s')
        return

    def get_df(self):
        if not os.path.exists(self.path_source):
            print('\nPath ' + self.path_source + ' does not exist.')
            print('Exiting program...\n')
            return False
        self.df = pd.read_csv(self.path_source)
        return True

    def set_path_dest(self):
        full_path = self.path_source.split('/')
        file_name = full_path[-1].split('_')[0]
        self.path_dest = self.dir_dest + '/' + file_name
        return 

    def split(self, test_size=0.3):
        X = self.select_columns(self.df, [slice(1, None)])
        y = self.df['rating']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
        self.df = None
        return 

    @timer_func
    def get_CV(self, name):
        name_train = '_' + name + '_train'
        name_test = '_' + name + '_test'
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.CV_thread(name_train)
            self.CV_thread(name_test)
            # t1 = threading.Thread(target=self.CV_thread, args=(name_train, ))
            # t2 = threading.Thread(target=self.CV_thread, args=(name_test, ))
            # t1.start()
            # t2.start()
            # t1.join()
            # t2.join()
        return 

    def CV_thread(self, name):
        if 'nltk' in name:
            cols = ['clean_nltk', 'sentiment_text', 'sentiment_nltk', ('Happy', 0)]
        elif 'spacy' in name:
            cols = ['clean_spacy', 'sentiment_text', 'sentiment_spacy', ('Happy', 0)]
        if 'train' in name: 
            X = self.X_train
            y = self.y_train
        elif 'test' in name:
            X = self.X_test
            y = self.y_test

        TFIDF=TfidfVectorizer(use_idf=True)
        y.name = 'rating'
        y = y.reset_index(drop=True)
        X = X.reset_index(drop=True)
        T = X[cols[0]]
        T = TFIDF.fit_transform(T)
        T = pd.DataFrame(T.todense(), columns=TFIDF.get_feature_names())
        X = self.select_columns(X, cols)
        T = pd.concat([y, X, T], axis=1)
        self.save_df(name, T)
        return 

    def save_df(self, name, df):
        name = self.path_dest + name + '.csv'
        df.to_csv(name)
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
    Setup(path_source=r'./output_2/450K_prep.csv', dir_dest=r'./output_3/')
    # Setup(path_source=r'./output_2/500_prep.csv', dir_dest=r'./output_3/')
    # Setup(path_source=r'./output_2/5K_prep.csv', dir_dest=r'./output_3/')
    

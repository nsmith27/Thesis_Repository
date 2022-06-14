#############################################################################################################################################
## Tasks                                                                                                                                    #
#############################################################################################################################################
# get csv data from ./output_1/
# create new column of cleaned text NLTK
# create a new column of cleaned text SPACY
# create column of sentiment from plain text
# create column of sentiment from clean_NLTK
# create column of sentiment from clean_SPACY
# create columns of emotion from plain text
# save dataframe in ./output_2/

def warn(*args, **kwargs):
    pass
from distutils.command.build_scripts import first_line_re
from pickle import TRUE
import warnings
warnings.warn = warn

import re
import os
import shutil
import warnings
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime, date
import pandas as pd
import spacy
import text2emotion as te
from string import punctuation
from autocorrect import Speller
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

print_log_path = './output_2/log.txt'
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
class Preprocess():
    # Before -- DF.columns
    #   {rating (int), review_text (str)} 
    # After  -- DF.columns
    #   {rating (int), review_text (str), clean_nltk (str), clean_spacy (str), 
    #    sentiment (float), sentiment_nltk (float), sentiment_spacy (float), 
    #    angry (float), fear (float), happy (float), sad (float), surpise (float)}

    def __init__(self, path_source=r'./output_2/450K_prep.csv', dir_dest=r'./output_2/'):
        self.analyzer = SentimentIntensityAnalyzer()
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.start_time = datetime.now().strftime("%H:%M")
        self.t0_class = time.time()
        # self.default_path_source = r'./output_1/450K_reviews.csv'
        # self.default_path_dest = r'./output_2/450K_prep.csv'
        self.path_source = path_source
        self.path_dest = dir_dest + path_source.split('/')[-1].replace('reviews', 'prep')
        self.df = None
        self.sentiment_read = None
        self.sentiment_write = None
        self.emotions = None
        self.set_stop_words = self.set_stop_words()

        if self.get_df():
            create_log(self.start_time, path_source, self.path_dest)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.wrap_clean('clean_nltk')
                self.wrap_clean('clean_spacy')
                self.wrap_sentiment('text', 10_000)
                self.wrap_sentiment('nltk', 10_000)
                self.wrap_sentiment('spacy', 10_000)    
                self.wrap_emotion(1000)

        return

    #############################################################################################################################################
    ## Helper Functions                                                                                                                         #
    #############################################################################################################################################
    def get_df(self):
        if not os.path.exists(self.path_source):
            print('\nPath ' + self.path_source + ' does not exist.')
            print('Exiting program...\n')
            return False
            # print('Path ' + self.path_source + ' does not exist.')
            # print('Using ' + self.default_path_source + ' as source...')
            # self.df = pd.read_csv(self.default_path_source, index_col=0)
        elif os.path.exists(self.path_dest):
            today = date.today()
            today = today.strftime("%b_%d_")
            full_path = self.path_dest.split('/')
            directory = full_path[:-1]
            file = '/' + today + full_path[-1]
            full_path = '/'.join(directory)+file
            print(self.path_dest, full_path)
            shutil.copyfile(self.path_dest, full_path)
            self.df = pd.read_csv(self.path_source, index_col=0)
        else:
            self.df = pd.read_csv(self.path_source, index_col=0)
        self.df.fillna('', inplace=True)
        return True

    def find_left_off(self, col, criterion):
        for i, v in enumerate(self.df[col]):
            if v == criterion:  
                break
        return i

    def save_df(self, include_time=False):
        print_log('\n...saving dataframe at ' + self.path_dest, '\t')
        self.df.to_csv(self.path_dest)
        if include_time:
            t2 = time.time()
            convert = time.strftime("%H:%M", time.gmtime(t2-self.t0_class))
            message = 'Start Time: ' + self.start_time + '\t' + f'\nTime since Preprocessing began: {convert} (Hrs:Min)'
            print_log(message)
        return

    ############################################################################################################################################
    ## Clean Text                                                                                                                              #
    # Normalizing case                                                                                                                         #
    # Remove extra line breaks                                                                                                                 #
    # Tokenize                                                                                                                                 #
    # Remove stop words and punctuations                                                                                                       #
    ############################################################################################################################################
    @timer_func
    def wrap_clean(self, new_col, chunk_size=500_000/12):
        func = None
        log_spacing = False
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
            first_miss = self.find_left_off(new_col, '')
            if first_miss >= size-1:
                if log_spacing:
                    print_log('\n')
                break
            else:
                log_spacing = True
            depth = min(first_miss + save_size, size)
            left_over = depth - first_miss
            chunk_size = math.ceil(left_over / num_cpu) if (chunk_size * num_cpu) > (left_over) else chunk_size
            input = [x[a:a+chunk_size] for a in range(first_miss, depth, chunk_size)]
            # Parallization
            pool = mp.Pool(mp.cpu_count())
            result = pool.map(func, input)
            pool.close()
            result = [item for sublist in result for item in sublist]
            self.df.loc[first_miss:depth, [new_col]] = result
            self.save_df(True)
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
        print_log(f'\n\tFunction \'clean_nltk\' executed in {(t2-t1):.4f}s')
        return out
    
    def clean_spacy(self, L):
        t1 = time.time()
        new_col = 'clean_spacy'
        out = []
        for i in range(len(L)):
            text = L[i]
            text = self.replace_web_reference(text)
            text = self.nlp(text)
            text = [word.lemma_ for word in text]
            text = self.normalize_case(text)
            text = self.remove_punc(text)
            text = self.remove_stop(text)
            text = ' '.join(text) if len(text) > 0 else L[i]
            out.append(text)
        t2 = time.time()
        print_log(f'\n\tFunction \'clean_spacy\' executed in {(t2-t1):.4f}s', '')
        return out

    def replace_web_reference(self, text):
        replacement = '-ref_website-'
        pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        try:
            text = re.sub(pattern, replacement, text)
        except:
            print_log('exception>>>>>>>>>>>')
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
        return [i for i in text if i not in self.stop_words]
    
    def set_stopwords(self):
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
        self.set_stop_words = stop_words
        return 

    #############################################################################################################################################
    ## Sentiment and Emotion                                                                                                                    #
    #############################################################################################################################################
    @timer_func
    def wrap_sentiment(self, read_col, chunk_size=500_000/12):
        log_spacing = False
        if 'text' in read_col:
            read_col = 'review_text'
            write_col = 'sentiment_text'
        elif 'nltk' in read_col:
            read_col = 'clean_nltk'
            write_col = 'sentiment_nltk'
        elif 'spacy' in read_col:
            read_col = 'clean_spacy'
            write_col = 'sentiment_spacy'
        else:
            return
        self.sentiment_read = read_col
        self.sentiment_write = write_col
        if write_col not in self.df:
            self.df[write_col] = -1

        num_cpu = mp.cpu_count()
        size = len(self.df.index)
        chunk_size = math.ceil(chunk_size)
        save_size = chunk_size * num_cpu
        x = list(self.df[read_col])
        while True:
            first_miss = self.find_left_off(write_col, -1)
            if first_miss >= size-1:
                if log_spacing:
                    print_log('\n')
                break
            else:
                log_spacing = True
            depth = min(first_miss + save_size, size)
            left_over = depth - first_miss
            chunk_size = math.ceil(left_over / num_cpu) if (chunk_size * num_cpu) > (left_over) else chunk_size
            input = [x[a:a+chunk_size] for a in range(first_miss, depth, chunk_size)]
            # Parallization
            pool = mp.Pool(mp.cpu_count())
            result = pool.map(self.get_sentiment, input)
            pool.close()
            result = [item for sublist in result for item in sublist]
            self.df.loc[first_miss:depth, [write_col]] = result
            self.save_df(True)
        return 

    def get_sentiment(self, L):
        t1 = time.time()
        if self.sentiment_write not in self.df:
            self.df[self.sentiment_write] = None
        out = []
        for text in L:
            score = self.analyzer.polarity_scores(text)['compound']
            score= (score + 1)/2
            out.append(score)
        t2 = time.time()
        print_log(f'\n\tFunction \'get_sentiment\' executed in {(t2-t1):.4f}s')
        return out

    @timer_func
    def wrap_emotion(self, chunk_size=500_000/12):
        log_spacing = False
        read_col = 'review_text'
        self.emotions = ('Happy', 'Angry', 'Surprise', 'Sad', 'Fear')
        for i in self.emotions:
            if i not in self.df:    
                self.df[i] = -1

        num_cpu = mp.cpu_count()
        size = len(self.df.index)
        chunk_size = math.ceil(chunk_size)
        save_size = chunk_size * num_cpu
        x = list(self.df[read_col])
        while True:
            fm = []
            for e in self.emotions:
                fm.append(self.find_left_off(e, -1))
            first_miss = min(fm)
            if first_miss >= size-1:
                if log_spacing:
                    print_log('\n')
                break
            else:
                log_spacing = True
            depth = min(first_miss + save_size, size)
            left_over = depth - first_miss
            chunk_size = math.ceil(left_over / num_cpu) if (chunk_size * num_cpu) > (left_over) else chunk_size
            input = [x[a:a+chunk_size] for a in range(first_miss, depth, chunk_size)]
            # Parallization
            pool = mp.Pool(mp.cpu_count())
            result = pool.map(self.get_emotion, input)
            pool.close()
            result = [item for sublist in result for item in sublist]
            result = np.transpose(result).tolist()
            for index, e in enumerate(self.emotions):
                self.df.loc[first_miss:depth, [e]] = result[index]
            self.save_df(True)
        return 
        
    def get_emotion(self, L):
        t1 = time.time()
        out = []
        for i in range(len(L)):
            text = L[i]
            row_emtns = te.get_emotion(text)
            temp = []
            for e in self.emotions:
                temp.append(row_emtns[e])
            out.append(temp)
        t2 = time.time()
        print_log(f'\n\tFunction \'get_emotion\' executed in {(t2-t1):.4f}s')
        return out

#############################################################################################################################################
## Main                                                                                                                                     #
#############################################################################################################################################
if __name__ == '__main__':
    mp.freeze_support()
    # Preprocess(path_source=r'./output_2/450K_prep.csv', dir_dest=r'./output_2/450K_prep.csv')
    # Preprocess(path_source=r'./output_1/462K_reviews.csv', dir_dest=r'./output_2/')
    # Preprocess(path_source=r'./output_2/462K_prep.csv', dir_dest=r'./output_2/')
    # Preprocess(path_source=r'./output_1/5K_reviews.csv', dir_dest=r'./output_2/')
    # Preprocess(path_source=r'./output_2/5K_reviews.csv', dir_dest=r'./output_2/')
    Preprocess(path_source=r'./output_1/500_reviews.csv', dir_dest=r'./output_2/')




#############################################################################################################################################
## Tasks                                                                                                                                     #
#############################################################################################################################################
# get raw text data from a path to /reviews/ directory 
# ...or get csv data from ./output_1/all_valid_reviews.csv
# select how many reviews of each rating you want
# ...ex: input n=90_000 means total of N=450K reviews
# ...ex: input n=100 means total of N=500 reviews
# save dataframe in ./output_1/N_reviews.csv

from distutils import text_file
import os
import re
import math
import time
import random
import pandas as pd
import multiprocessing as mp

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from autocorrect import Speller
from string import punctuation



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
class Collect():
    # Before -- reviews in text files
    # After  -- DF.columns {rating (int), review_text (str)}

    def __init__(self, dir_source=r'./reviews/', dir_dest=r'./output_1/', count=-1, save=True):
        if count >= 0:
            dir_source = r'./output_1/all_valid_reviews.csv'
        if not os.path.exists(dir_source):
            return
        self.save = save
        self.dir_source = dir_source
        self.review_paths = None
        self.out_dir = dir_dest
        self.out_file = r'reviews.csv'
        self.num_each = count
        self.D = {'1':[], '2':[], '3':[], '4':[], '5':[]}
       
        return

    @timer_func
    def convert_store_raw_data(self): 
        if self.num_each >= 0 and os.path.exists(self.dir_source):
            self.read_csv_reviews(self.dir_source)
        else: 
            self.read_parallel()
        self.check_alter_size()
        if self.num_each >= 0:
            for key in self.D.keys():
                kept = random.sample(self.D[key], self.num_each)
                self.D[key] = kept
        if self.save:
            self.save_to_csv()
        return 

    def read_csv_reviews(self, stored_path, col='review_text'):
        self.D = {'1':[], '2':[], '3':[], '4':[], '5':[]}
        df = pd.read_csv(stored_path)
        df.fillna('', inplace=True)
        for i in self.D:
            self.D[i] = df.query('rating == ' + str(i))[col].astype('string').tolist()
            self.D[i] = ['\"' + k.replace('\"', "\'") + '\"'for k in self.D[i]]
        return 

    def read_parallel(self):
        self.get_paths()
        # Parallization setup
        num_cpu = mp.cpu_count()
        size = len(self.review_paths)
        chunk_size = math.ceil(size/num_cpu)
        input = [self.review_paths[i:i+chunk_size] for i in range(0, len(self.review_paths), chunk_size)]
        # Parallization start
        pool = mp.Pool(num_cpu)
        result = pool.map(self.parse_review, input)
        pool.close()
        # Parallization end
        list_tup = [item for sublist in result for item in sublist]
        for t in list_tup:
            self.D[t[0]].append(t[1])

        return 

    def get_paths(self):
        ftypes = '.txt'
        paths = os.listdir(self.dir_source)
        self.review_paths = [(self.dir_source + i) for i in paths if ftypes == i[-4:] ]
        return 

    def parse_review(self, L):
        t1 = time.time()
        result = []
        for path in L:
            content = self.read_file(path)
            content = content.split('Review:')
            delimiter=''
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
        t2 = time.time()
        print(f'Function \'parse_review\' executed in {(t2-t1):.4f}s')
        return result

    def read_file(self, path):
        result = ''
        with open(path, 'r', encoding='utf8') as file:
            result = file.read()
        return result 

    def check_alter_size(self):
        x = min([len(self.D[key]) for key in self.D])
        if x < self.num_each:
            self.num_each = x
            print('Input number too large.\nValue was changed to ' + str(x))
        return 

    def save_to_csv(self):
        name = ''
        if self.num_each < 0:
            name = 'all_' + self.out_file
        elif self.num_each*5 < 1000:
            name = str(self.num_each*5) + '_' + self.out_file
        else:
            name = str(int(self.num_each/200)) + 'K_' + self.out_file
        path = self.out_dir + name
        with open(path, 'w', encoding='utf-8') as file:
            columns =  'rating,review_text\n'
            file.write(columns)
            for n in self.D:
                for k in range(len(self.D[n])):
                    input = n + ',' + self.D[n][k] + '\n'
                    file.write(input)
        return

    ############################################################################################################################################
    ## Exploratory Functionality                                                                                                               #
    ############################################################################################################################################
    @timer_func
    def generate_word_stats(self, col=None, WC=False):
        if col:
            path = './output_1/words_spell_clean.csv'
            self.generate_clean_spellcheck(path)
            self.read_csv_reviews(path, col)
            for key in self.D:
                text = '\n'.join(self.D[key])
                text = re.sub('[^\s0-9a-zA-Z]+', '', text).split()
                text = FreqDist(text)
                text.plot(50,title='Frequency distribution for 30 most common tokens in ' + str(key) + '-star reviews.')
        else:
            path = r'./output_1/all_valid_reviews.csv'
            self.read_csv_reviews(path)
        if WC:
            for key in self.D:
                text = '\n'.join(self.D[key])
                wordcloud = WordCloud(colormap='hsv').generate(text)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()
        self.D = {'1':[], '2':[], '3':[], '4':[], '5':[]}
        return

    @timer_func
    def generate_clean_spellcheck(self, path):
        if not os.path.exists(path):
            source = './output_1/462K_reviews.csv'
            self.df = pd.read_csv(source, usecols = ['rating', 'review_text'])
            self.wrap_clean(path, 'spell_clean')
        return 

    def wrap_clean(self, out_path, new_col, chunk_size=500_000/12):
        if new_col not in self.df:
            self.df[new_col] = ''
        num_cpu = mp.cpu_count()
        size = len(self.df.index)
        chunk_size = math.ceil(chunk_size)
        save_size = chunk_size * num_cpu
        x = list(self.df['review_text'])
        while True:
            t1 = time.time()
            first_miss = self.find_left_off(new_col, '')
            if first_miss >= size-1:
                break
            depth = min(first_miss + save_size, size)
            left_over = depth - first_miss
            chunk_size = math.ceil(left_over / num_cpu) if (chunk_size * num_cpu) > (left_over) else chunk_size
            input = [x[a:a+chunk_size] for a in range(first_miss, depth, chunk_size)]
            # Parallization
            pool = mp.Pool(mp.cpu_count())
            result = pool.map(self.clean_nltk, input)
            pool.close()
            result = [item for sublist in result for item in sublist]
            self.df.loc[first_miss:depth, [new_col]] = result
            # Save and report
            self.df.to_csv(out_path, index=False)        
            t2 = time.time()
            print(f'\n\tFunction \'clean_nltk\' executed in {(t2-t1):.4f}s')
        return 

    def find_left_off(self, col, criterion):
        for i, v in enumerate(self.df[col]):
            if v == criterion:  
                break
        return i

    def clean_nltk(self, L):
        out = []
        for i in range(len(L)):
            text = L[i]
            text = re.sub('[^\s0-9a-zA-Z]+', '', text)
            text = self.replace_web_reference(text)
            text = word_tokenize(text)
            text = [t.lower() for t in text] 
            # text = self.spell_correct(text)
            text = self.remove_stop(text)
            text = ' '.join(text) if len(text) > 0 else L[i]
            out.append(text)
        return out

    def remove_stop(self, text):
        stop_words = stopwords.words("english")
        more_words = '''
        I'm
        I
        it
        it's
        would
        could
        also
        no
        not
        did
        don
        dont
        don't
        won
        wont
        won't
        recipie
        shouldn
        shouldnt
        shouldn't
        too
        '''.split('\n')[1:-1]
        stop_words = stop_words + more_words
        return [i for i in text if i.lower() not in stop_words]

    def replace_web_reference(self, text):
        replacement = '-ref_website-'
        pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        try:
            text = re.sub(pattern, replacement, text)
        except:
            pass
        return text
    
    def spell_correct(self, text):
        spell = Speller(lang='en')
        for i, v in enumerate(text):
            if v == 'ref_website':
                continue
            text[i] = spell(v)

        return text


############################################################################################################################################
## Main                                                                                                                                    #
############################################################################################################################################
if __name__ == '__main__':
    mp.freeze_support()
    # Collect(dir_source=r'C:/Users/natha/Desktop/2022 Thesis/Thesis Code/reviews/' , count=100)
    # Collect() 
    # explore = Collect(count=400, save=False)
    # explore.generate_clean_spellcheck()
    explore = Collect(save=False) 
    explore.generate_word_stats(col='spell_clean', WC=True)


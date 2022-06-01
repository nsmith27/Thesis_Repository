#############################################################################################################################################
## Tasks                                                                                                                                     #
#############################################################################################################################################
# get raw text data from a path to /reviews/ directory 
# ...or get csv data from ./output_1/all_valid_reviews.csv
# select how many reviews of each rating you want
# ...ex: input n=90_000 means total of N=450K reviews
# ...ex: input n=100 means total of N=500 reviews
# save dataframe in ./output_1/N_reviews.csv

import os
import re
import math
import time
import random
import pandas as pd
import multiprocessing as mp


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

    def __init__(self, dir_source=r'./reviews/', dir_dest=r'./output_1/', count=-1):
        if count >= 0:
            dir_source = r'./output_1/all_valid_reviews.csv'
        if not os.path.exists(dir_source):
            return
        self.dir_source = dir_source
        self.review_paths = None
        self.out_dir = dir_dest
        self.out_file = r'reviews.csv'
        self.num_each = count

        self.convert_store_raw_data()
        return

    @timer_func
    def convert_store_raw_data(self): 
        D = {'1':[], '2':[], '3':[], '4':[], '5':[]}
        stored_path = self.dir_source
        if self.num_each >= 0 and os.path.exists(stored_path):
            df = pd.read_csv(stored_path)
            df.fillna('', inplace=True)
            for i in D:
                D[i] = df.query('rating == ' + str(i))['review_text'].astype('string').tolist()
                D[i] = ['\"' + k.replace('\"', "\'") + '\"'for k in D[i]]
        else: 
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
                D[t[0]].append(t[1])
        self.check_alter_size(D)
        if self.num_each >= 0:
            for key in D.keys():
                kept = random.sample(D[key], self.num_each)
                D[key] = kept
        self.save_to_csv(D)
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

    def check_alter_size(self, D):
        x = min([len(D[key]) for key in D])
        if x < self.num_each:
            self.num_each = x
            print('Input number too large.\nValue was changed to ' + str(x))
        return 

    def save_to_csv(self, D):
        name = ''
        if self.num_each < 0:
            name = 'all_' + self.out_file
        elif self.num_each < 1000:
            name = str(self.num_each*5) + '_' + self.out_file
        else:
            name = str(int(self.num_each/200)) + 'K_' + self.out_file
        path = self.out_dir + name
        with open(path, 'w', encoding='utf-8') as file:
            columns =  'rating,review_text\n'
            file.write(columns)
            for n in D:
                for k in range(len(D[n])):
                    input = n + ',' + D[n][k] + '\n'
                    file.write(input)
        return


############################################################################################################################################
## Main                                                                                                                                    #
############################################################################################################################################
if __name__ == '__main__':
    mp.freeze_support()
    # Collect(dir_source=r'C:/Users/natha/Desktop/2022 Thesis/Thesis Code/reviews/' , count=100)
    Collect(count=100)

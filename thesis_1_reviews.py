import chunk
import os
import re
import math
import time
import random
import multiprocessing as mp


############################################################################################################################################
## Get Data 
# Raw Data reader 
# Store all in csv file
############################################################################################################################################
class Collect():
    # Before -- reviews in text files
    # After  -- DF.columns {rating (int), review_text (str)}

    def __init__(self, dir_source=r'./reviews/', dir_dest=r'./output_1/', count=-1):
        if not os.path.exists(dir_source):
            return

        self.dir_source = dir_source
        self.review_paths = None
        self.out_dir = dir_dest
        self.out_file = r'reviews.txt'
        self.num_each = count

        self.get_paths()
        self.convert_store_raw_data()
        return

    def get_paths(self):
        ftypes = '.txt'
        paths = os.listdir(self.dir_source)
        self.review_paths = [(self.dir_source + i) for i in paths if ftypes == i[-4:] ]
        return 

    def convert_store_raw_data(self): 
        D = {'1':[], '2':[], '3':[], '4':[], '5':[]}
        # Parallization
        num_cpu = mp.cpu_count()
        size = len(self.review_paths)
        chunk_size = math.ceil(size/num_cpu)
        input = [self.review_paths[i:i+chunk_size] for i in range(0, len(self.review_paths), chunk_size)]
        pool = mp.Pool(num_cpu)
        result = pool.map(self.parse_review, input)
        pool.close()
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

    def read_file(self, path):
        result = ''
        with open(path, 'r', encoding='utf8') as file:
            result = file.read()
        return result 

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
            name = str(self.num_each) + '_' + self.out_file
        else:
            name = str(self.num_each/1000) + 'K_' + self.out_file
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
## Descriptive/Exploratory Data Analysis 
# Document term matrix using TfIdf
############################################################################################################################################
if __name__ == '__main__':
    mp.freeze_support()
    Collect(count=90_000)


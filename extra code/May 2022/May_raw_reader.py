import os
import re


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

# convert_store_raw_data(path_450K, 90_000)

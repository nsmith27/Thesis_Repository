import os 


# max_ratings = 0
# path = r'C:/Users/natha/Desktop/recipes'

# list = os.listdir(path)
# p1 = path + '/668.txt'
# text = ''

# count = 0
# L = {}
# for i, v in enumerate(list):
#     with open(path+'/' + v, 'r', encoding='utf-8') as f:
#         text = f.read().split('\n')
#     x = 'Number of ratings:'
#     try:
#         x = text.index(x)
#     except:
#         continue
#     x = text[x+1].split()[0]
#     if x.isnumeric():
#         x = int(x)
#         if x in L:
#             L[x] += 1
#         else:
#             L[x] = 1
#         if int(x) > max_ratings:
#             max_ratings = int(x)
#             count += 1
#             print(max_ratings, i, count)

# import matplotlib.pyplot as plt

# names = [*L.keys()]
# values = [*L.values()]

# plt.bar(range(len(L)), values)
# plt.show()


# # defining the libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# # No of Data points
# N = 500
# # initializing random values
# data = np.random.randn(N)
# # getting data of the histogram
# count, bins_count = np.histogram(L, bins=10)
# # finding the PDF of the histogram using count values
# pdf = count / sum(count)
# # using numpy np.cumsum to calculate the CDF
# # We can also find using the PDF values by looping and adding
# cdf = np.cumsum(pdf)
# # plotting PDF and CDF
# plt.plot(bins_count[1:], pdf, color="red", label="PDF")
# plt.plot(bins_count[1:], cdf, label="CDF")
# plt.legend()

# max_ratings = 0
# path = r'C:/Users/natha/Desktop/recipe_reviews'

# list = os.listdir(path)
# text = ''

# count = 0
# D = dict()
# for i, v in enumerate(list):
#     with open(path+'/' + v, 'r', encoding='utf-8') as f:
#         text = f.read().split('\n')
#     L = [text[i+1] for i, x in enumerate(text) if x == 'Display Name:']
#     for x in L:
#         if x.strip() == '':
#             continue
#         if x in D:
#             D[x] += 1
#         else:
#             D[x] = 1
    

# import pandas as pd
# # df = pd.DataFrame.from_dict(D, orient='index') 
# # df.to_csv('getter.csv')


# df = pd.read_csv('getter.csv', index_col=[0])
# # df = df.sort_values(by=['num_reviews'], ascending=False)
# # df.to_csv('getter2.csv')
# print(df.index.size)
# print(df.head())

more_words = '''
like
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
shouldn
shouldnt
shouldn't
too
recipe
recipes
'''.split('\n')[1:-1]
print(more_words)
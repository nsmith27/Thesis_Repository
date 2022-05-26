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

import numpy as np

a = [1,2]
b = []
c = ['a','b']

b.insert(0,a)
b.insert(-1,c)
print(b)

x = np.transpose(b).tolist()
print(x)


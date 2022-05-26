

# #############################################################################################################################################
# ## Main                                                                                                                                     #
# #############################################################################################################################################
# @timer_func
# def run(source, N, bin='5', tool='NLTK', save=True):
#     print()
#     path_N = str(int(N*5/1000)) + r'K_reviews.csv'
#     path_no_extension = str(int(N*5/1000)) + r'K'
#     df = pd.read_csv(source)
#     df = select_rows_by_rating(df, N)
#     # Bin review column or not
#     bin = str(bin)
#     special = ''
#     if bin == '2a':
#         df = reduce_2Abin(df)
#         special = '2Abin'
#     elif bin == '2b':
#         df = reduce_2Bbin(df)
#         special = '2Bbin'
#     elif bin == '3a':
#         df = reduce_3Abin(df)
#         special = '3Abin'
#     elif bin == '3b':
#         df = reduce_3Bbin(df)
#         special = '3Bbin'
#     else:
#         special = '5bin'

#     # Choose which library/package to use
#     tool = tool.lower()
#     if tool == 'nltk':
#         path_no_extension = path_no_extension + '_NLTK_' + special
#         df = clean_NLTK(df, path_no_extension, save)
#     elif tool == 'spacy':
#         path_no_extension = path_no_extension + '_SPACY_' + special
#         df = clean_spaCy(df, path_no_extension, save)

#     X = select_columns(df, [(2,0)])
#     y = df['rating']
#     TextID(X, y, path_no_extension)
#     return 

# @timer_func
# def main():
#     path_all = r'all_reviews.csv'
#     path_all_valid = r'all_valid_reviews.csv'
#     path_450K = r'450K_reviews.csv'
#     # convert_store_raw_data(path_450K, 90_000)

#     N = 10
#     run(path_450K, N, '2a', 'NLTK')
#     run(path_450K, N, '2b', 'NLTK')
#     run(path_450K, N, '3a', 'NLTK')
#     run(path_450K, N, '3b', 'NLTK')
#     run(path_450K, N, '5', 'NLTK')

#     run(path_450K, N, '2a', 'SPACY')
#     run(path_450K, N, '2b', 'SPACY')
#     run(path_450K, N, '3a', 'SPACY')
#     run(path_450K, N, '3b', 'SPACY')
#     run(path_450K, N, '5', 'SPACY')

#     return 


# if __name__ == "__main__":
#     main()




# from thesis_1_reviews import *
from thesis_2_features import *
# from thesis_3_ML import *
# from thesis_4_analyze import *


# convert_store_raw_data(path_450K, 90_000)
Preprocess()




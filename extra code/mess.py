# https://towardsdatascience.com/review-rating-prediction-a-combined-approach-538c617c495c
# https://medium.com/analytics-vidhya/predicting-the-ratings-of-reviews-of-a-hotel-using-machine-learning-bd756e6a9b9b
# https://github.com/hsinyuchen1017/Python-Prediction-of-Rating-using-Review-Text/blob/master/Code-Hsin-Yu%20Chen.ipynb


import re
from string import punctuation
import pandas as pd, numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize       
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


############################################################################################################################################
### Text clean up 
############################################################################################################################################
reviews0 = pd.read_csv("Zomato_reviews.csv")
reviews0.head()
reviews0.describe(include="all")
reviews1 = reviews0[~reviews0.review_text.isnull()].copy()
reviews1.reset_index(inplace=True, drop=True)
reviews0.shape, reviews1.shape


############################################################################################################################################
### Normalizing case
############################################################################################################################################
reviews_list = reviews1.review_text.values
len(reviews_list)
reviews_lower = [txt.lower() for txt in reviews_list]


############################################################################################################################################
### Remove extra line breaks
############################################################################################################################################
reviews_lower[2:4]
reviews_lower = [" ".join(txt.split()) for txt in reviews_lower]
reviews_lower[2:4]


############################################################################################################################################
#### Tokenize
############################################################################################################################################
print(word_tokenize(reviews_lower[0]))
reviews_tokens = [word_tokenize(sent) for sent in reviews_lower]
print(reviews_tokens[0])


############################################################################################################################################
### Remove stop words and punctuations
############################################################################################################################################
stop_nltk = stopwords.words("english")
stop_punct = list(punctuation)
print(stop_nltk)
stop_nltk.remove("no")
stop_nltk.remove("not")
stop_nltk.remove("don")
stop_nltk.remove("won")
"no" in stop_nltk
stop_final = stop_nltk + stop_punct + ["...", "``","''", "====", "must"]
def del_stop(sent):
    return [term for term in sent if term not in stop_final]
del_stop(reviews_tokens[1])
reviews_clean = [del_stop(sent) for sent in reviews_tokens]
reviews_clean = [" ".join(sent) for sent in reviews_clean]
reviews_clean[:2]


############################################################################################################################################
### Separate X and Y and perform train test split, 70-30
############################################################################################################################################
len(reviews_clean)
X = reviews_clean
y = reviews1.rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)


############################################################################################################################################
### Document term matrix using TfIdf
############################################################################################################################################
vectorizer = TfidfVectorizer(max_features = 5000)
len(X_train), len(X_test)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)
X_train_bow.shape, X_test_bow.shape


############################################################################################################################################
### Model building
############################################################################################################################################
?RandomForestRegressor
learner_rf = RandomForestRegressor(random_state=42)
%%time
learner_rf.fit(X_train_bow, y_train)
y_train_preds = learner_rf.predict(X_train_bow)
mean_squared_error(y_train, y_train_preds)**0.5


############################################################################################################################################
#### Increase the number of trees
############################################################################################################################################
learner_rf = RandomForestRegressor(random_state=42, n_estimators=20)
%%time
learner_rf.fit(X_train_bow, y_train)
y_train_preds = learner_rf.predict(X_train_bow)
mean_squared_error(y_train, y_train_preds)**0.5


############################################################################################################################################
### Hyper-parameter tuning
############################################################################################################################################
?RandomForestRegressor
learner_rf = RandomForestRegressor(random_state=42)
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_features': [500, "sqrt", "log2", "auto"],
    'max_depth': [10, 15, 20, 25]
}
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = learner_rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 1, scoring = "neg_mean_squared_error" )
grid_search.fit(X_train_bow, y_train)
grid_search.grid_scores_
grid_search.best_estimator_


############################################################################################################################################
### Using the best estimator to make predictions on the test set
############################################################################################################################################
y_train_pred = grid_search.best_estimator_.predict(X_train_bow)
y_test_pred = grid_search.best_estimator_.predict(X_test_bow)
mean_squared_error(y_train, y_train_pred)**0.5
mean_squared_error(y_test, y_test_pred)**0.5


############################################################################################################################################
### Identifying mismatch cases
############################################################################################################################################
res_df = pd.DataFrame({'review':X_test, 'rating':y_test, 'rating_pred':y_test_pred})
res_df[(res_df.rating - res_df.rating_pred)>=2].shape
res_df[(res_df.rating - res_df.rating_pred)>=2]

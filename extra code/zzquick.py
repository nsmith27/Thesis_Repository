import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# from nltk.tokenize import word_tokenize 
# from nltk.stem import WordNetLemmatizer


# # sentence_data = "Sun rises in the east. Sun sets in the west."
# # nltk_tokens = nltk.sent_tokenize(sentence_data)
# # # print (nltk_tokens)
# word_data = "It originated !!from ! the! idea? \"that there are readers who prefer\" learning new skills from the comforts of their drawing rooms"
# # nltk_tokens = nltk.word_tokenize(word_data)
# # print (nltk_tokens)

# # Set up for text cleaning
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer() 
# word_list = []
# text_data = word_data
# jo = ''
# # Cleaning process
# for i in range(1):
#     td = word_data
#     text = str(td)
#     text = text.lower()
#     tokens = word_tokenize(text)
#     words = [word for word in tokens if word.isalpha()]
#     words = [w for w in words if not w in stop_words]
#     clean_words = [lemmatizer.lemmatize(word) for word in words]
#     for cw in clean_words:
#         word_list.append(cw)
#     cleanedList = [x for x in word_list if x != 'nan']
#     jo = " ".join(cleanedList)
#     word_list[:] = []
#     cleanedList[:] = []

# print(word_list)
# # print(cleanedList)




# stop_words = stopwords.words("english")
# keep = '''
# no
# not
# don
# dont
# won
# wont
# shouldn
# shouldnt
# too
# '''.split('\n')[1:-1]
# print(keep)
# # print(stop_words)

import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
a = '''
This cookies were just blehhhh.  Not the brownie-type batter I was expecting.  I followed the recipe exactly, too.  I think they might be better with margarine or butter instead of the shortening.
Way too salty not at all what I was wanting. I will rinse it and see if I can salvage it at all.
this recipe was bland and didn't form a good dough. what a waste of ingredients.
Dry and flavorless! I've no doubt it's on account of the 1/2 cup of cornstarch. I've just wasted 2 sticks of butter on this recipe. NOT worth it.
I could not get this recipe to work, I have 2 cookie press and they work fine with my other recipe. It seems like its to wet.
We found the taste and consistency of these cookies to be just awful. I think the half cup of cornstarch was just too much.
This recipe is awful. The cookies were tasteless.
This recipe was really disappointing. I was really excited to use my cookie press for the first time. These cookies formed really nicely but when I took them out and sampled one, they tasted awful. They were bitter and tasted like flour. I had even added a bit more sugar than called for.
I've had lot better cookies
I didn't care for this recipe.  I followed the instructions and it turned out to be very dry.   I added water and it was so thick it broke my cookie press.  Maybe I should have used melted butter.
These came out very dry  and crumbly.
I followed the recipe to a T and the cookies came out bland and crumbly. So the next batch I added just a bit of sherry to make the dough more pliable and sprinkled chopped pecans on top.   Still bland and crumbly.
I followed the recipe exactly and it was very dry and bland -no flavor whatsoever . My family didn't like them either.  The texture of the dough was ok-a little odd, but worked with my press.  Very chalky cookie though-I will not be making this again.
I tried this recipe today.  Sorry to say all I could taste was the corn starch, and my mouth immediatly went dry.  My neighbor tried one and didn't care for it either.
Tried this recipe to go on a chocolate cake I had baked it I did not like it at all. It had this extra sweet (from the honey) but tangy flavor (from the cream cheese) that just didn't take good to me.
This was a waste of my time energy and money. I think the recipe was not complete.
So bland  , would not make these again without adding a whole lot of spices, but then that would be a whole different recipe.        Emeril's jalapeno poppers are awesome, a lot more work but at least they are edible.
'''.split('\n')

# texts = '''
# here is line 1
# this is another sentence
# dogs are not cats
# your mom sucks!
# '''.split('\n')
docs = nlp.pipe(a, n_process=4)

# for i in docs:
#     print(i)

# exit()
for i, v in enumerate(docs):
    text = v
    # text = nlp(text)
    text = [word.lemma_ for word in text]
    text = ' '.join(text)
    print(text)
    print()

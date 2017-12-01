
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
import pickle, csv, random
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')


# In[2]:

dataframes = {
    "cooking": pd.read_csv("/home/neo/ml1/data/light/cooking_light.csv"),
    "crypto": pd.read_csv("/home/neo/ml1/data/light/crypto_light.csv"),
    "robotics": pd.read_csv("/home/neo/ml1/data/light/robotics_light.csv"),
    "biology": pd.read_csv("/home/neo/ml1/data/light/biology_light.csv"),
    "travel": pd.read_csv("/home/neo/ml1/data/light/travel_light.csv"),
    "diy": pd.read_csv("/home/neo/ml1/data/light/diy_light.csv"),
    "physics": pd.read_csv("/home/neo/ml1/data/light/physics_light.csv"),
}


# In[3]:

physics_index = dataframes["cooking"].shape[0]+dataframes["crypto"].shape[0]+dataframes["robotics"].shape[0]                    +dataframes["biology"].shape[0]+dataframes["travel"].shape[0]+dataframes["diy"].shape[0]
print(dataframes["physics"].shape[0]+physics_index,physics_index)


# In[4]:

new_df = pd.DataFrame()
train_df = pd.DataFrame()
for df in dataframes:
    new_df = new_df.append(dataframes[df])
    new_df.fillna("NAN")
train_df["content"] = new_df["title"] +" "+ new_df["content"]
corpus = []
for row in train_df["content"]:
    corpus.append(str(row))


# In[ ]:

tfidf_matrix =  tf.fit_transform(corpus)
feature_names = tf.get_feature_names() 


# In[ ]:

dense=[]
for i in range(tfidf_matrix.shape[0]):
    temp = tfidf_matrix[i].todense().tolist()[0]
    dense += temp
    if(i%1000 == 0):
        print("Step {0} out of {1}".format(i, tfidf_matrix.shape[0]), end="\r", flush=True)
    print("Length of dense is: {0}".format(len(dense)))
    del temp


# In[ ]:

dense_file = open("/home/neo/ml1/data/light/dense.pkl","wb")
pickle.dump(dense, dense_file)


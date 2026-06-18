import os
import kagglehub
import pandas as pd
import numpy as np 
import pickle
from statistics import mode 
import nltk
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import json

os.environ['KAGGLE_USERNAME'] = 'kausthubhdarbha' 
os.environ['KAGGLE_KEY'] = 'KGAT_b194539e027e8c84efaa3a36cf906407'

print("Loading Amazon Fine Food Reviews...")

path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
csv_path = os.path.join(path, "Reviews.csv")

# 3. Load the data ( only read the first 50,000 rows)
df = pd.read_csv(csv_path, nrows=50000)

df.drop_duplicates(subset = ['Text'], inplace = True)
df.dropna(axis = 0, inplace = True)
input_data = df.loc[:, 'Text']
target_data = df.loc[:,'Summary']
target_data.replace('', np.nan, inplace = True)

input_texts = []
target_texts = []
input_words = []
target_words = []
with open("contractions.json", "r") as f:
    contractions_dict = json.load(f)
stop_words = set(stopwords.words('english'))
stemm = LancasterStemmer()

def clean(texts,src):
    #remove html tags
    texts = BeautifulSoup(texts, "lxml").text.lower()
    #contraction seperation using dictionary
    for word, expansion in contractions_dict.items():
        texts = texts.replace(word, expansion)

    words = word_tokenize(texts.lower())
    words = list(filter(lambda w: (w.isalpha() and len(w) >= 3), words))
    
    #stem words to get roots and filter stop words
    if src == "inputs":
        words = [stemm.stem(w) for w in words if w not in stop_words]
    else:
        words = [w for w in words if w not in stop_words]
    return words

#pass input and target records 
for in_txt, tr_txt in zip(input_data, target_data):
    in_words = clean(in_txt, "inputs")
    input_texts += [' '.join(in_words)]
    input_words += in_words

    #add 'sos' and 'eos' at start and end of text
    tr_words = clean(tr_txt, "target")
    tr_words = ['sos'] + tr_words + ['eos']
    target_texts += [' '.join(tr_words)]
    target_words += tr_words

#store unique words from input and target list of words
input_words = sorted(list(set(input_words)))
target_words = sorted(list(set(target_words)))
num_in_words = len(input_words)
num_tr_words = len(target_words)

#get length of the input and target texts which appear the most
max_in_len = mode([len(i) for i in input_texts])
max_tr_len = mode([len(i) for i in target_texts])

print("number of input words : ",num_in_words)
print("number of target words : ",num_tr_words)
print("maximum input length : ",max_in_len)
print("maximum target length : ",max_tr_len)

print(f"Loaded {len(df)} reviews into the dataframe.")

#creating train test splits (80 - 20)
x_train, x_test, y_train, y_test = train_test_split(input_texts, target_texts, test_size=0.2, random_state=0)


# ==========================================
# YOUR ML TEXT SUMMARIZATION CODE GOES HERE
# ==========================================

# Let's see what we are working with
#print("\nSample Data:")
#print(df[['Summary', 'Text']].head())

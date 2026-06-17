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

# 1. Quietly ensure Kaggle knows who you are (Safe to leave this)
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username' 
os.environ['KAGGLE_KEY'] = 'KGAT_b194539e027e8c84efaa3a36cf906407'

print("Loading Amazon Fine Food Reviews...")

# 2. This instantly locates the files on your Mac (No redownloading!)
path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
csv_path = os.path.join(path, "Reviews.csv")

# 3. Load the data (Let's only read the first 50,000 rows so your Mac doesn't lag!)
# The full dataset has over 500,000 rows!
df = pd.read_csv(csv_path, nrows=50000)

df.drop_duplicates(subset = ['Text'], inplace = True)
df.dropna(axis = 0, inplace = True)
input_data = df.loc[:, 'Text']
target_data = df.loc[:,'Summary']
target.replace('', np.nan, inplace = True)

print(f"Loaded {len(df)} reviews into the dataframe.")



# ==========================================
# YOUR ML TEXT SUMMARIZATION CODE GOES HERE
# ==========================================

# Let's see what we are working with
print("\nSample Data:")
print(df[['Summary', 'Text']].head())

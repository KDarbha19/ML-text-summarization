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
df = df.head(10000)

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

#train the tokenizer with all the words
in_tokenizer = Tokenizer()
in_tokenizer.fit_on_texts(x_train)
tr_tokenizer = Tokenizer()
tr_tokenizer.fit_on_texts(y_train)

num_in_words = len(in_tokenizer.word_index)
num_tr_words = len(tr_tokenizer.word_index)

#convert text into seq of integers, interger represents index of word
x_train = in_tokenizer.texts_to_sequences(x_train)
y_train = tr_tokenizer.texts_to_sequences(y_train)

#pad array of 0's if length is less than max length
en_in_data = pad_sequences(x_train, maxlen = max_in_len, padding = 'post')
dec_data = pad_sequences(y_train, maxlen = max_tr_len, padding = 'post')

#decoder input data != last word(eos) decoder target data != first word(sos)
dec_in_data = dec_data[:,:-1]
dec_tr_sliced = dec_data[:, 1:]
dec_tr_data = dec_tr_sliced.reshape(len(dec_data), max_tr_len - 1,1)

K.clear_session()
latent_dim = 256

#create input objects of total number of encoder words
en_inputs = Input(shape=(max_in_len,))
en_embedding = Embedding(num_in_words+1, latent_dim)(en_inputs)

#create 3 stacked LSTM layers 

#LSTM 1
en_lstml = LSTM(latent_dim, return_state = True, return_sequences = True)
en_outputs1, state_h1, state_c1 = en_lstml(en_embedding)

#LSTM 2
en_lstm2 = LSTM(latent_dim, return_state = True, return_sequences = True)
en_outputs2, state_h2, state_c2 = en_lstm2(en_outputs1)

#LSTM 3
en_lstm3 = LSTM(latent_dim, return_sequences = True, return_state = True)
en_outputs3, state_h3, state_c3 = en_lstm3(en_outputs2)

#encoder states
en_states = [state_h3, state_c3]

# Decoder. 
dec_inputs = Input(shape=(None,)) 
dec_emb_layer = Embedding(num_tr_words+1, latent_dim) 
dec_embedding = dec_emb_layer(dec_inputs) 
 
#initialize decoder's LSTM layer with the output states of encoder
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_outputs, *_ = dec_lstm(dec_embedding,initial_state=en_states)

#Attention layer
attention =Attention()
attn_out = attention([dec_outputs,en_outputs3])
 
#Concatenate the attention output with the decoder outputs
merge=Concatenate(axis=-1, name='concat_layer1')([dec_outputs,attn_out])

#Dense layer (output layer)
dec_dense = Dense(num_tr_words+1, activation='softmax') 
dec_outputs = dec_dense(merge) 

#Model class and model summary 
model = Model([en_inputs, dec_inputs], dec_outputs)
model.summary()
#plot_model(model, to_file = 'model_plot.png', show_shapes = True, show_layer_names = True)

model.compile(optimizer = "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(
    [en_in_data, dec_in_data], 
    dec_tr_data, 
    batch_size = 256, 
    epochs = 10, 
    validation_split=0.1,
)

model.save("s2s.keras")

#encoder inference
latent_dim = 256

#load model
model = models.load_model("variables/s2s.keras", custom_objects={"Attention": Attention})

#construct encoder model from the output of 6 later
en_outputs3 = model.get_layer("lstm_2").output[0] #3rd LSTM layer
state_h_enc = model.get_layer("lstm_2").output[1] #Final hidden atate
state_c_enc = model.get_layer("lstm_2").output[2] #Final cell state

en_states = [state_h_enc,state_c_enc]

#add input and state from the layer
en_model = Model(model.input[0], [en_outputs3, state_h_enc, state_c_enc])

#decoder inference
dec_inputs = model.input[1]
dec_emb_layer = model.get_layer("embedding_1")
dec_embedding = dec_emb_layer(dec_inputs)

#define states 
dec_state_input_h = Input(shape=(latent_dim,), name = "decoder_state_h")
dec_state_input_c = Input(shape=(latent_dim,), name = "decoder_state_c")
dec_hidden_state_input = Input(shape=(max_in_len, latent_dim))
dec_states_inputs = [dec_state_input_h, dec_state_input_c]

#Decode LSTM layer 
dec_lstm = model.get_layer("lstm_3")
dec_outputs2, state_h2,state_c2 = dec_lstm(dec_embedding, initial_state = dec_states_inputs)

#Attention Layer
attention = model.get_layer("attention")
attn_out2 = attention([dec_outputs2, dec_hidden_state_input])

#concatenate layer
concat_layer = model.get_layer("concat_layer1")
merge2 = concat_layer([dec_outputs2, attn_out2])

#Dense Layer
dec_dense = model.get_layer("dense")
dec_outputs2 = dec_dense(merge2)

#Defining model class
dec_model = Model(
    [dec_inputs] + [dec_hidden_state_input, dec_state_input_h, dec_state_input_c],
    [dec_outputs2] + [state_h2, state_c2]
)



#create a dictionary with a key as index and value as words.
reverse_target_word_index = tr_tokenizer.index_word
reverse_source_word_index = in_tokenizer.index_word
target_word_index = tr_tokenizer.word_index
reverse_target_word_index[0]=' '
 
def decode_sequence(input_seq):
    #get the encoder output and states by passing the input sequence
    en_out, en_h, en_c= en_model.predict(input_seq, verbose = 0)
 
    #target sequence with initial word as 'sos'
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['sos']
 
    #if the iteration reaches the end of text than it will be stop the iteration
    stop_condition = False
    #append every predicted word in decoded sentence
    decoded_sentence = ""
    while not stop_condition: 
        #get predicted output, hidden and cell state.
        output_words, dec_h, dec_c= dec_model.predict([target_seq] + [en_out,en_h, en_c], verbose = 0)
        
        #get the index and from the dictionary get the word for that index.
        word_index = np.argmax(output_words[0, -1, :])
        text_word = reverse_target_word_index[word_index]

        if text_word == 'eos':
            stop_condition = True
            break
        
        if word_index != 0:
            decoded_sentence += text_word + " "

        if len(decoded_sentence.split()) >= max_tr_len:
            stop_condition = True
            break
        
        target_seq = np.zeros((1,1))
        target_seq[0,0] = word_index
        en_h, en_c = dec_h, dec_c
    
    return decoded_sentence.strip()


# ==========================================
# YOUR ML TEXT SUMMARIZATION CODE GOES HERE
# ==========================================

# Let's see what we are working with
#print("\nSample Data:")
#print(df[['Summary', 'Text']].head())

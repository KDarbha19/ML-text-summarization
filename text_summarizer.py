import os
import kagglehub
import pandas as pd
import numpy as np 
import json
from statistics import mode 
import nltk
from nltk import word_tokenize
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
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

os.environ['KAGGLE_USERNAME'] = 'kausthubhdarbha' 
os.environ['KAGGLE_KEY'] = 'KGAT_b194539e027e8c84efaa3a36cf906407'

print("Loading Amazon Fine Food Reviews...")
path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
csv_path = os.path.join(path, "Reviews.csv")

df = pd.read_csv(csv_path, nrows=120000)

pos_df = df[df['Score'] >= 4].head(15000)
neg_df = df[df['Score'] <= 2].head(15000)

print("Total balanced rows available after filtering (Pos):", len(pos_df))
print("Total balanced rows available after filtering (Neg):", len(neg_df))
df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=0).reset_index(drop=True)

df.drop_duplicates(subset=['Text'], inplace=True)
df.dropna(axis=0, inplace=True)
input_data = df.loc[:, 'Text']
target_data = df.loc[:, 'Summary']

input_texts = []
target_texts = []

with open("contractions.json", "r") as f:
    contractions_dict = json.load(f)
stop_words = set(stopwords.words('english'))

# FIX 1: Removed stemming entirely — stemmed inputs produce unreadable summaries
# and create a token space mismatch between encoder and decoder
def clean(texts):
    texts = BeautifulSoup(texts, "lxml").text.lower()
    for word, expansion in contractions_dict.items():
        texts = texts.replace(word, expansion)
    words = word_tokenize(texts)
    words = [w for w in words if w.isalpha() and len(w) >= 3]
    words = [w for w in words if w not in stop_words]
    return words

for in_txt, tr_txt in zip(input_data, target_data):
    if type(tr_txt) != str:
        continue
    in_words = clean(in_txt)
    input_texts.append(' '.join(in_words))

    tr_words = clean(tr_txt)
    tr_words = ['sos'] + tr_words + ['eos']
    target_texts.append(' '.join(tr_words))

eighty_in_len = int(np.percentile([len(i.split()) for i in input_texts], 80))
eighty_tr_len = int(np.percentile([len(i.split()) for i in target_texts], 80))
max_in_len = max(eighty_in_len, 35)
max_tr_len = max(eighty_tr_len, 15)

print("maximum input word length : ", max_in_len)
print("maximum target word length : ", max_tr_len)

x_train, x_test, y_train, y_test = train_test_split(input_texts, target_texts, test_size=0.2, random_state=0)

# FIX 2: Cap vocab size so the output Dense layer isn't projecting over 50k+ words
in_tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
in_tokenizer.fit_on_texts(x_train)
tr_tokenizer = Tokenizer(num_words=8000, oov_token='<OOV>')
tr_tokenizer.fit_on_texts(y_train)

num_in_words = min(len(in_tokenizer.word_index), 20000)
num_tr_words = min(len(tr_tokenizer.word_index), 8000)

print("Encoder vocab size:", num_in_words)
print("Decoder vocab size:", num_tr_words)

x_train_seq = in_tokenizer.texts_to_sequences(x_train)
y_train_seq = tr_tokenizer.texts_to_sequences(y_train)

en_in_data = pad_sequences(x_train_seq, maxlen=max_in_len, padding='post')
dec_data = pad_sequences(y_train_seq, maxlen=max_tr_len, padding='post')

dec_in_data = dec_data[:, :-1]
dec_tr_sliced = dec_data[:, 1:]
dec_tr_data = dec_tr_sliced.reshape(len(dec_data), max_tr_len - 1, 1)

K.clear_session()
latent_dim = 128

# --- ENCODER ---
en_inputs = Input(shape=(max_in_len,), name="encoder_inputs")
en_embedding = Embedding(num_in_words + 1, latent_dim, mask_zero=True, name="encoder_emb")(en_inputs)

en_lstm1 = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.3, name="enc_lstm_1")
en_outputs1, _, _ = en_lstm1(en_embedding)

en_lstm2 = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.3, name="enc_lstm_2")
en_outputs2, _, _ = en_lstm2(en_outputs1)

en_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, name="enc_lstm_3")
en_outputs3, state_h3, state_c3 = en_lstm3(en_outputs2)
en_states = [state_h3, state_c3]

# --- DECODER ---
# FIX 3: Removed recurrent_dropout from decoder LSTM — re-enables CuDNN kernel,
# cutting epoch time roughly in half with no meaningful accuracy cost
dec_inputs = Input(shape=(None,), name="decoder_inputs")
dec_emb_layer = Embedding(num_tr_words + 1, latent_dim, mask_zero=True, name="decoder_emb")
dec_embedding = dec_emb_layer(dec_inputs)

dec_lstm_layer = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, name="decoder_lstm")
dec_outputs, *_ = dec_lstm_layer(dec_embedding, initial_state=en_states)

# --- ATTENTION & OUTPUT ---
attention = Attention(name="attention_layer")
attn_out = attention([dec_outputs, en_outputs3])

merge = Concatenate(axis=-1, name='concat_layer1')([dec_outputs, attn_out])
dec_dense = Dense(num_tr_words + 1, activation='softmax', name="dense_layer")
dec_outputs = dec_dense(merge)

model = Model([en_inputs, dec_inputs], dec_outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# FIX 4: Added callbacks — stops training when val_loss plateaus, saves best weights,
# and reduces LR when stuck (your previous run was overfitting from epoch 14 onward)
os.makedirs("variables", exist_ok=True)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ModelCheckpoint('variables/best_model.keras', save_best_only=True, monitor='val_loss', verbose=1)
]

model.fit(
    [en_in_data, dec_in_data], dec_tr_data,
    batch_size=256,
    epochs=25,
    validation_split=0.1,
    callbacks=callbacks
)

model.save("variables/s2s.keras")
print("\n--- MODEL TRAINED & SAVED SUCCESSFULLY. STARTING INFERENCE ---")

# Load the best checkpoint (not necessarily the last epoch)
model = models.load_model("variables/best_model.keras", custom_objects={"Attention": Attention})

# FIX 5: Correct inference model construction — use model.input[0/1] instead of
# model.get_layer("...").input which returns empty after load_model in Keras 3
encoder_input_tensor = model.input[0]
en_lstm3_layer = model.get_layer("enc_lstm_3")
en_out_seq = en_lstm3_layer.output[0]
en_state_h = en_lstm3_layer.output[1]
en_state_c = en_lstm3_layer.output[2]
en_model = Model(encoder_input_tensor, [en_out_seq, en_state_h, en_state_c])

decoder_input_tensor = model.input[1]
dec_emb_layer = model.get_layer("decoder_emb")
dec_embedding_inf = dec_emb_layer(decoder_input_tensor)

dec_state_input_h = Input(shape=(latent_dim,), name="inf_decoder_state_h")
dec_state_input_c = Input(shape=(latent_dim,), name="inf_decoder_state_c")
dec_hidden_state_input = Input(shape=(max_in_len, latent_dim), name="inf_decoder_hidden_states")

dec_lstm_inf = model.get_layer("decoder_lstm")
dec_out_inf, state_h_inf, state_c_inf = dec_lstm_inf(
    dec_embedding_inf,
    initial_state=[dec_state_input_h, dec_state_input_c]
)

attention_layer_inf = model.get_layer("attention_layer")
attn_out_inf = attention_layer_inf([dec_out_inf, dec_hidden_state_input])

concat_layer_inf = model.get_layer("concat_layer1")
merge_inf = concat_layer_inf([dec_out_inf, attn_out_inf])

dec_dense_inf = model.get_layer("dense_layer")
dec_final_out = dec_dense_inf(merge_inf)

dec_model = Model(
    [decoder_input_tensor, dec_hidden_state_input, dec_state_input_h, dec_state_input_c],
    [dec_final_out, state_h_inf, state_c_inf]
)

reverse_target_word_index = tr_tokenizer.index_word
target_word_index = tr_tokenizer.word_index
reverse_target_word_index[0] = ' '

# FIX 6: Decoder state bug fixed — dec_h/dec_c update each step,
# en_out stays fixed as the encoder context throughout generation
def decode_sequence(en_out, en_h, en_c):
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['sos']
    dec_h, dec_c = en_h, en_c  # initialize decoder state from encoder final state

    decoded_sentence = ""
    while True:
        output_words, dec_h, dec_c = dec_model.predict(
            [target_seq, en_out, dec_h, dec_c], verbose=0
        )

        preds = output_words[0, -1, :]
        temperature = 0.7
        preds = np.log(preds + 1e-10) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        word_index = np.random.choice(range(len(preds)), p=preds)
        text_word = reverse_target_word_index.get(word_index, '')

        if text_word == 'eos' or len(decoded_sentence.split()) >= (max_tr_len - 1):
            break

        if word_index != 0 and text_word not in ('sos', '<OOV>'):
            decoded_sentence += text_word + " "

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = word_index

    return decoded_sentence.strip()

# Interactive loop
while True:
    inp_review = input("\nEnter Review (or 'quit' to exit): ")
    if inp_review.lower() == 'quit':
        break

    print("Processing...")
    inp_review_cleaned = clean(inp_review)
    inp_review_joined = ' '.join(inp_review_cleaned)

    inp_x = in_tokenizer.texts_to_sequences([inp_review_joined])
    inp_x = pad_sequences(inp_x, maxlen=max_in_len, padding='post')

    en_out, en_h, en_c = en_model.predict(inp_x.reshape(1, max_in_len), verbose=0)
    summary = decode_sequence(en_out, en_h, en_c)

    print("Predicted summary:", summary)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


data = pd.read_csv('English2Hindi.csv', encoding='utf-8').dropna(subset=['English', 'Hindi'])
# Keep short sentence pairs (<= 10 words) and a 10k sample so padding and training stay manageable
eng_len = data['English'].str.split().str.len()
hin_len = data['Hindi'].str.split().str.len()
data = data[(eng_len <= 10) & (hin_len <= 10)].sample(n=10000, random_state=42).reset_index(drop=True)

X_data = data['English'].astype(str).str.lower().str.strip()
y_data = data['Hindi'].astype(str).str.strip().apply(lambda x: 'startseq ' + x + ' endseq')

eng_tokenizer = Tokenizer(filters='')
eng_tokenizer.fit_on_texts(X_data)
num_enc_toks = len(eng_tokenizer.word_index) + 1

hin_tokenizer = Tokenizer(filters='')
hin_tokenizer.fit_on_texts(y_data)
num_dec_toks = len(hin_tokenizer.word_index) + 1

enc_input_sequences = eng_tokenizer.texts_to_sequences(X_data)
dec_sequences = hin_tokenizer.texts_to_sequences(y_data)

max_len_src = max(len(seq) for seq in enc_input_sequences)
max_len_tar = max(len(seq) for seq in dec_sequences)

encoder_input_data = pad_sequences(enc_input_sequences, maxlen=max_len_src, padding='post')
padded_dec_sequences = pad_sequences(dec_sequences, maxlen=max_len_tar, padding='post')

decoder_input_data = padded_dec_sequences[:, :-1]
decoder_output_data = padded_dec_sequences[:, 1:]


max_len_tar = max_len_tar - 1

latent_dim = 250

enc_inps = Input(shape=(max_len_src,))
enc_emb = Embedding(num_enc_toks, latent_dim, mask_zero=True)(enc_inps)
enc_outputs, st_h, st_c = LSTM(latent_dim, return_state=True)(enc_emb)
enc_states = [st_h, st_c]

dec_inps = Input(shape=(max_len_tar,))
dec_emb = Embedding(num_dec_toks, latent_dim, mask_zero=True)(dec_inps)
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_outputs, _, _ = dec_lstm(dec_emb, initial_state=enc_states)
dec_dense = Dense(num_dec_toks, activation='softmax')
dec_outputs = dec_dense(dec_outputs)

model = Model([enc_inps, dec_inps], dec_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


print("Training model...")
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_output_data,
    batch_size=16,
    epochs=50,
    validation_split=0.2
)


enc_model = Model(enc_inps, enc_states)

dec_st_inp_h = Input(shape=(latent_dim,))
dec_st_inp_c = Input(shape=(latent_dim,))
dec_states_inps = [dec_st_inp_h, dec_st_inp_c]

dec_emb2 = Embedding(num_dec_toks, latent_dim, mask_zero=True)(dec_inps)

dec_emb2 = model.layers[3](dec_inps)

dec_outputs2, st_h2, st_c2 = model.layers[5](dec_emb2, initial_state=dec_states_inps)
dec_outputs2 = model.layers[6](dec_outputs2)

dec_model = Model([dec_inps] + dec_states_inps, [dec_outputs2] + [st_h2, st_c2])

rev_tar_char_idx = {i: word for word, i in hin_tokenizer.word_index.items()}

def translate(input_text):
    seq = eng_tokenizer.texts_to_sequences([input_text])
    padded_seq = pad_sequences(seq, maxlen=max_len_src, padding='post')

    states_value = enc_model.predict(padded_seq, verbose=0)
    tar_seq = np.zeros((1, max_len_tar))
    tar_seq[0, 0] = hin_tokenizer.word_index['startseq']

    dec_sen = []

    for t in range(max_len_tar - 1):
        output_toks, h, c = dec_model.predict([tar_seq] + states_value, verbose=0)
        sampled_tok_idx = np.argmax(output_toks[0, t, :])

        if sampled_tok_idx == 0:
            break

        sampled_char = rev_tar_char_idx.get(sampled_tok_idx, '')

        if sampled_char == 'endseq':
            break

        dec_sen.append(sampled_char)
        tar_seq[0, t + 1] = sampled_tok_idx

    return ' '.join(dec_sen)

print("\n--- Translation Test ---")
for i in range(0,5):
  test_phrase = X_data.iloc[i]
  print(f"English Input: {test_phrase}")
  print(f"Predicted Hindi: {translate(test_phrase)}")
  print(f"Actual Expected Hindi: {data['Hindi'].iloc[i]}")

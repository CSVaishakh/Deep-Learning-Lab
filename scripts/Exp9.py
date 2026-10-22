import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv("English2Hindi.csv").dropna(subset=["English", "Hindi"])
data["English"] = data["English"].astype(str)
data["Hindi"] = data["Hindi"].astype(str)

# Keep short sentences to make training faster
data = data[
    (data["English"].str.split().str.len() <= 20) &
    (data["Hindi"].str.split().str.len() <= 20)
].reset_index(drop=True)


# Build word dictionaries
eng_words = sorted(set(" ".join(data["English"]).split()))
hin_words = sorted(set(" ".join(data["Hindi"]).split()))

eng_to_id = {word: i + 1 for i, word in enumerate(eng_words)}
hin_to_id = {word: i + 1 for i, word in enumerate(hin_words)}
id_to_hin = {i: word for word, i in hin_to_id.items()}

max_eng_len = int(data["English"].str.split().str.len().max())
max_hin_len = int(data["Hindi"].str.split().str.len().max())
eng_vocab_size = len(eng_to_id) + 1  # 0 is padding
hin_vocab_size = len(hin_to_id) + 1

X_train, X_test, y_train, y_test = train_test_split(
    data["English"], data["Hindi"], test_size=0.2, random_state=42
)



# Convert sentences into padded integer arrays
def encode_text(text, word_to_id, max_len):
    result = np.zeros(max_len, dtype="int32")
    for i, word in enumerate(text.split()[:max_len]):
        result[i] = word_to_id.get(word, 0)
    return result

def make_arrays(X, y):
    enc = np.array([encode_text(s, eng_to_id, max_eng_len) for s in X])
    dec_in = np.array([encode_text(s, hin_to_id, max_hin_len) for s in y])
    dec_out = np.zeros((len(y), max_hin_len), dtype="int32")
    for i, sentence in enumerate(y):
        ids = [hin_to_id.get(w, 0) for w in sentence.split()[:max_hin_len]]
        if len(ids) > 1:
            dec_out[i, :len(ids)-1] = ids[1:]
    return enc, dec_in, dec_out

Xtr, Ytr, Ytr_out = make_arrays(X_train.tolist(), y_train.tolist())
Xte, Yte, Yte_out = make_arrays(X_test.tolist(), y_test.tolist())



# Encoder-decoder model
latent_dim = 128

enc_input = Input(shape=(max_eng_len,))
enc_embedding = Embedding(eng_vocab_size, latent_dim, mask_zero=True)(enc_input)
_, enc_h, enc_c = LSTM(latent_dim, return_state=True)(enc_embedding)
enc_states = [enc_h, enc_c]

dec_input = Input(shape=(max_hin_len,))
dec_embedding_layer = Embedding(hin_vocab_size, latent_dim, mask_zero=True)
dec_embedding = dec_embedding_layer(dec_input)
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_output, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)
dec_dense = Dense(hin_vocab_size, activation="softmax")
dec_output = dec_dense(dec_output)

model = Model([enc_input, dec_input], dec_output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()



# Train only as long as validation loss keeps improving
early_stop = EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)

history = model.fit(
    [Xtr, Ytr], Ytr_out,
    validation_data=([Xte, Yte], Yte_out),
    batch_size=128,
    epochs=10,                 # maximum; EarlyStopping may stop sooner
    callbacks=[early_stop],
    verbose=1
)



# Inference models
encoder_model = Model(enc_input, enc_states)

state_h = Input(shape=(latent_dim,))
state_c = Input(shape=(latent_dim,))
decoder_states = [state_h, state_c]

step_embedding = dec_embedding_layer(dec_input)
step_output, out_h, out_c = dec_lstm(
    step_embedding, initial_state=decoder_states
)
step_output = dec_dense(step_output)
decoder_model = Model(
    [dec_input, state_h, state_c],
    [step_output, out_h, out_c]
)

def translate(sentence):
    encoded = encode_text(sentence, eng_to_id, max_eng_len)[None, :]
    h, c = encoder_model.predict(encoded, verbose=0)
    result = []

    # Start from the first Hindi word ID available; this dataset version
    # does not define explicit START_/END_ tokens.
    current_id = 0
    for _ in range(max_hin_len):
        dec = np.zeros((1, max_hin_len), dtype="int32")
        dec[0, 0] = current_id
        probs, h, c = decoder_model.predict([dec, h, c], verbose=0)
        current_id = int(np.argmax(probs[0, 0]))
        if current_id == 0:
            break
        result.append(id_to_hin.get(current_id, ""))
    return " ".join(result)

sample = X_test.iloc[0]
print("English:", sample)
print("Predicted Hindi:", translate(sample))
print("Actual Hindi:", y_test.iloc[0])



# Translate a sentence from the training set
k = 0
hin_sen = translate(X_train.iloc[k])
print(f'''Input English sentence: {X_train.iloc[k]}\n
          Predicted Hindi Translation: {hin_sen}\n
          Actual Hindi Translation: {y_train.iloc[k]}''')

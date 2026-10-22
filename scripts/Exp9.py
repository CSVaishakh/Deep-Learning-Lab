import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# On Colab, upload English2Hindi.csv when asked
if not os.path.exists("English2Hindi.csv"):
    from google.colab import files
    files.upload()

data = pd.read_csv("English2Hindi.csv").dropna(subset=["English", "Hindi"])
print("Rows:", len(data))


MAX_WORDS = 20      # keep sentences with at most this many words
ENG_VOCAB = 15000   # keep the most frequent words; the rest become <unk>
HIN_VOCAB = 15000

def clean(text):
    text = str(text).lower().strip()
    text = re.sub(r"([?.!,।|])", r" \1 ", text)  # split punctuation from words
    return re.sub(r"\s+", " ", text).strip()

data["English"] = data["English"].map(clean)
data["Hindi"] = data["Hindi"].map(clean)

data = data[
    data["English"].str.split().str.len().between(1, MAX_WORDS) &
    data["Hindi"].str.split().str.len().between(1, MAX_WORDS)
].drop_duplicates(subset=["English"]).reset_index(drop=True)

print("Sentence pairs used:", len(data))
data[["English", "Hindi"]].head()


# Special tokens: 0 = padding, 1 = unknown word, 2 = start, 3 = end
PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"

X_train, X_test, y_train, y_test = train_test_split(
    data["English"], data["Hindi"], test_size=0.1, random_state=42
)

def build_vocab(sentences, size):
    counts = Counter(w for s in sentences for w in s.split())
    words = [PAD, UNK, START, END] + [w for w, _ in counts.most_common(size - 4)]
    return {w: i for i, w in enumerate(words)}

eng_to_id = build_vocab(X_train, ENG_VOCAB)
hin_to_id = build_vocab(y_train, HIN_VOCAB)
id_to_hin = {i: w for w, i in hin_to_id.items()}

max_eng_len = MAX_WORDS
max_hin_len = MAX_WORDS + 1  # room for <start> / <end>

def encode(sentence, vocab):
    return [vocab.get(w, vocab[UNK]) for w in sentence.split()]

def make_arrays(X, y):
    enc = [encode(s, eng_to_id) for s in X]
    hin = [encode(s, hin_to_id) for s in y]
    # Teacher forcing: decoder reads "<start> w1 w2 ..." and predicts "w1 w2 ... <end>"
    dec_in = [[hin_to_id[START]] + h for h in hin]
    dec_out = [h + [hin_to_id[END]] for h in hin]
    pad = lambda seqs, n: keras.utils.pad_sequences(seqs, maxlen=n, padding="post")
    return pad(enc, max_eng_len), pad(dec_in, max_hin_len), pad(dec_out, max_hin_len)

Xtr, Ytr_in, Ytr_out = make_arrays(X_train, y_train)
Xte, Yte_in, Yte_out = make_arrays(X_test, y_test)

print("English vocab:", len(eng_to_id), "| Hindi vocab:", len(hin_to_id))
print("Train:", Xtr.shape, "| Test:", Xte.shape)


# Encoder-decoder LSTM with attention
EMB_DIM = 256
UNITS = 512

# Encoder
enc_input = keras.Input(shape=(None,), name="english")
enc_emb = layers.Embedding(len(eng_to_id), EMB_DIM, mask_zero=True)(enc_input)
enc_outputs, enc_h, enc_c = layers.LSTM(
    UNITS, return_sequences=True, return_state=True
)(enc_emb)

# Decoder (layers kept as variables so the inference model can reuse them)
dec_input = keras.Input(shape=(None,), name="hindi_in")
dec_emb_layer = layers.Embedding(len(hin_to_id), EMB_DIM, mask_zero=True)
dec_lstm = layers.LSTM(UNITS, return_sequences=True, return_state=True)
attention = layers.Attention()
concat = layers.Concatenate()
dec_hidden = layers.Dense(UNITS, activation="tanh")
dec_dense = layers.Dense(len(hin_to_id), activation="softmax")

dec_seq, _, _ = dec_lstm(dec_emb_layer(dec_input), initial_state=[enc_h, enc_c])
context = attention([dec_seq, enc_outputs])  # which English words to look at
dec_output = dec_dense(dec_hidden(concat([dec_seq, context])))

model = keras.Model([enc_input, dec_input], dec_output)

# Per-word accuracy. Keras skips padded positions automatically because
# the decoder embedding uses mask_zero=True (same for the loss).
def word_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    return tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1)), tf.float32)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[word_accuracy]
)
model.summary()


early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    [Xtr, Ytr_in], Ytr_out,
    validation_data=([Xte, Yte_in], Yte_out),
    batch_size=128,
    epochs=30,           # maximum; EarlyStopping usually stops sooner
    callbacks=[early_stop],
    verbose=2            # one line per epoch
)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="validation")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["word_accuracy"], label="train")
ax[1].plot(history.history["val_word_accuracy"], label="validation")
ax[1].set_title("Word accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.show()


# Inference models: run the encoder once, then the decoder one word at a time
encoder_model = keras.Model(enc_input, [enc_outputs, enc_h, enc_c])

step_input = keras.Input(shape=(1,))
enc_out_input = keras.Input(shape=(None, UNITS))
enc_mask_input = keras.Input(shape=(None,), dtype="bool")
h_input = keras.Input(shape=(UNITS,))
c_input = keras.Input(shape=(UNITS,))

step_seq, step_h, step_c = dec_lstm(
    dec_emb_layer(step_input), initial_state=[h_input, c_input]
)
step_context = attention([step_seq, enc_out_input], mask=[None, enc_mask_input])
step_probs = dec_dense(dec_hidden(concat([step_seq, step_context])))

decoder_model = keras.Model(
    [step_input, enc_out_input, enc_mask_input, h_input, c_input],
    [step_probs, step_h, step_c]
)

def translate_batch(sentences, batch_size=256):
    results = []
    for b in range(0, len(sentences), batch_size):
        chunk = sentences[b:b + batch_size]
        enc = keras.utils.pad_sequences(
            [encode(clean(s), eng_to_id)[:max_eng_len] for s in chunk],
            maxlen=max_eng_len, padding="post"
        )
        enc_out, h, c = encoder_model(enc, training=False)
        enc_mask = tf.constant(enc != 0)
        token = np.full((len(chunk), 1), hin_to_id[START])
        predicted = []
        for _ in range(max_hin_len):
            probs, h, c = decoder_model(
                [tf.constant(token), enc_out, enc_mask, h, c], training=False
            )
            token = np.argmax(np.asarray(probs)[:, -1, :], axis=-1)[:, None]
            predicted.append(token[:, 0])
        for row in np.stack(predicted, axis=1):
            words = []
            for t in row:
                if t == hin_to_id[END]:
                    break
                words.append(id_to_hin.get(int(t), UNK))
            results.append(" ".join(words))
    return results

def translate(sentence):
    return translate_batch([sentence])[0]

k = 0
print(f'''Input English sentence: {X_train.iloc[k]}\n
          Predicted Hindi Translation: {translate(X_train.iloc[k])}\n
          Actual Hindi Translation: {y_train.iloc[k]}''')


# Translations of sentences the model has never seen
samples = X_test.iloc[:10].tolist()
for eng, pred, actual in zip(samples, translate_batch(samples), y_test.iloc[:10]):
    print("English  :", eng)
    print("Predicted:", pred)
    print("Actual   :", actual)
    print()


# BLEU score on 1000 test sentences (0-100, higher is better)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

n = 1000
predictions = translate_batch(X_test.iloc[:n].tolist())
references = [[ref.split()] for ref in y_test.iloc[:n]]
hypotheses = [p.split() for p in predictions]
bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
print(f"BLEU score: {bleu * 100:.2f}")

# Import libraries
import tensorflow as tf
import pandas as pd
import time

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense

# 1. Load IMDB Dataset
vocab_size = 10000
max_length = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

# 2. Preprocess Dataset
x_train = pad_sequences(
    x_train, maxlen=max_length,
    padding="post", truncating="post"
)

x_test = pad_sequences(
    x_test, maxlen=max_length,
    padding="post", truncating="post"
)

# 3. Build Model
def build_model(architecture):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, mask_zero=True))

    if architecture == "RNN":
        model.add(SimpleRNN(64))
    elif architecture == "LSTM":
        model.add(LSTM(64))
    else:
        model.add(GRU(64))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 4. Train and Evaluate Models
architectures = ["RNN", "LSTM", "GRU"]
results = []

for architecture in architectures:

    print("\nTraining model:", architecture)

    model = build_model(architecture)

    start_time = time.time()

    model.fit(
        x_train,
        y_train,
        epochs=6,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    training_time = time.time() - start_time

    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, verbose=0
    )

    results.append({
        "Model": architecture,
        "Test Accuracy (%)": round(test_accuracy * 100, 2),
        "Test Loss": round(test_loss, 4),
        "Training Time (seconds)": round(training_time, 2)
    })

# 5. Display Comparison Table
results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("          RNN, LSTM AND GRU PERFORMANCE COMPARISON")
print("=" * 75)

print(
    f"{'Model':<15}"
    f"{'Test Accuracy (%)':>20}"
    f"{'Test Loss':>15}"
    f"{'Training Time (s)':>23}"
)

print("-" * 75)

for _, row in results_df.iterrows():
    print(
        f"{row['Model']:<15}"
        f"{row['Test Accuracy (%)']:>20.2f}"
        f"{row['Test Loss']:>15.4f}"
        f"{row['Training Time (seconds)']:>23.2f}"
    )

print("=" * 75)

# 6. Final Analysis
best_accuracy_model = results_df.loc[
    results_df["Test Accuracy (%)"].idxmax(), "Model"
]

lowest_loss_model = results_df.loc[
    results_df["Test Loss"].idxmin(), "Model"
]

print("\n" + "=" * 35)
print("          FINAL ANALYSIS")
print("=" * 35)
print(f"  Best Test Accuracy : {best_accuracy_model}")
print(f"  Lowest Test Loss   : {lowest_loss_model}")
print("=" * 35)

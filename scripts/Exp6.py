import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import time


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Add channel dimension
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Resize images from 28x28 to 32x32
x_train = tf.image.resize(x_train, (32, 32))
x_test = tf.image.resize(x_test, (32, 32))

# Convert grayscale images to RGB
x_train = tf.image.grayscale_to_rgb(x_train)
x_test = tf.image.grayscale_to_rgb(x_test)

# VGG19 preprocessing
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


base_model = VGG19(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3)
)

# Freeze all VGG19 layers
base_model.trainable = False

# Add classifier
x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation="relu")(x)
output = layers.Dense(10, activation="softmax")(x)

model_fixed = Model(
    inputs=base_model.input,
    outputs=output
)

# Compile model
model_fixed.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

start = time.time()

history_fixed = model_fixed.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

fixed_time = time.time() - start


# Evaluate fixed model
fixed_loss, fixed_acc = model_fixed.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n====================================")
print("FIXED FEATURE EXTRACTION")
print("====================================")

print("Test Accuracy :", fixed_acc * 100, "%")
print("Test Loss :", fixed_loss)
print("Training Time :", fixed_time, "seconds")

base_model_ft = VGG19(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3)
)

# Allow VGG19 to be trainable
base_model_ft.trainable = True

# Freeze most layers
for layer in base_model_ft.layers[:-4]:
    layer.trainable = False

# Add classifier
x = layers.Flatten()(base_model_ft.output)
x = layers.Dense(512, activation="relu")(x)
output = layers.Dense(10, activation="softmax")(x)

model_ft = Model(
    inputs=base_model_ft.input,
    outputs=output
)

# Compile with a small learning rate
model_ft.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

start = time.time()

history_ft = model_ft.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

fine_time = time.time() - start

# Evaluate fine-tuned model
fine_loss, fine_acc = model_ft.evaluate(
    x_test,
    y_test,
    verbose=0
)


print("\n================================")
print("FINE-TUNING")
print("================================")

print("Test Accuracy :", fine_acc * 100, "%")
print("Test Loss :", fine_loss)
print("Training Time :", fine_time, "seconds")

print("\n================================")
print("PERFORMANCE COMPARISON")
print("================================")

print(
    "Fixed Feature Extraction Accuracy :",
    fixed_acc * 100, "%"
)

print(
    "Fine-Tuning Accuracy             :",
    fine_acc * 100, "%"
)

print(
    "Fixed Feature Extraction Loss    :",
    fixed_loss
)

print(
    "Fine-Tuning Loss                 :",
    fine_loss
)

print(
    "Fixed Feature Extraction Time    :",
    fixed_time,
    "seconds"
)

print(
    "Fine-Tuning Time                 :",
    fine_time,
    "seconds"
)


# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(
    history_fixed.history["accuracy"],
    marker="o",
    label="Training Accuracy"
)
axes[0, 0].plot(
    history_fixed.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_title(
    "Fixed Feature Extraction - Accuracy"
)
axes[0, 0].legend()
axes[0, 0].grid()

axes[0, 1].plot(
    history_fixed.history["loss"],
    marker="o",
    label="Training Loss"
)
axes[0, 1].plot(
    history_fixed.history["val_loss"],
    marker="o",
    label="Validation Loss"
)
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].set_title(
    "Fixed Feature Extraction - Loss"
)
axes[0, 1].legend()
axes[0, 1].grid()

axes[1, 0].plot(
    history_ft.history["accuracy"],
    marker="o",
    label="Training Accuracy"
)
axes[1, 0].plot(
    history_ft.history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].set_title("Fine-Tuning - Accuracy")
axes[1, 0].legend()
axes[1, 0].grid()

axes[1, 1].plot(
    history_ft.history["loss"],
    marker="o",
    label="Training Loss"
)
axes[1, 1].plot(
    history_ft.history["val_loss"],
    marker="o",
    label="Validation Loss"
)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].set_title("Fine-Tuning - Loss")
axes[1, 1].legend()
axes[1, 1].grid()

# Adjust spacing
plt.tight_layout()
plt.show()

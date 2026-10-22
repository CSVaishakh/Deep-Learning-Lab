import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical



(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("================================")
print("          MODEL 1")
print("================================")

model1 = models.Sequential([
    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64, (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])

model1.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model1.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

loss1, accuracy1 = model1.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n================================")
print("       MODEL 1 RESULTS")
print("================================")
print(f"Test Accuracy : {accuracy1 * 100:.2f}%")
print(f"Test Loss     : {loss1:.4f}")

indices = [10, 20]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i, index in enumerate(indices):

    image = X_test[index]

    prediction = model1.predict(
        image.reshape(1, 28, 28, 1),
        verbose=0
    )

    predicted_digit = np.argmax(prediction)
    actual_digit = np.argmax(y_test[index])
    one_hot_vector = y_test[index]

    axes[i].imshow(
        image.reshape(28, 28),
        cmap="gray"
    )

    axes[i].set_title(
        f"Model 1 - Example {i + 1}",
        fontsize=12
    )

    axes[i].axis("off")

    result_text = (
        f"Actual Digit   : {actual_digit}\n"
        f"One-Hot Vector : {one_hot_vector.astype(int)}\n"
        f"Predicted Digit: {predicted_digit}"
    )

    axes[i].text(
        0.5,
        -0.25,
        result_text,
        transform=axes[i].transAxes,
        ha="center",
        va="top",
        fontsize=10
    )

plt.tight_layout()
plt.show()


print("\n\n================================")
print("          MODEL 2")
print("================================")

model2 = models.Sequential([
    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64, (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        10,
        activation="softmax"
    )
])

model2.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model2.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

loss2, accuracy2 = model2.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n================================")
print("       MODEL 2 RESULTS")
print("================================")
print(f"Test Accuracy : {accuracy2 * 100:.2f}%")
print(f"Test Loss     : {loss2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i, index in enumerate(indices):

    image = X_test[index]

    prediction = model2.predict(
        image.reshape(1, 28, 28, 1),
        verbose=0
    )

    predicted_digit = np.argmax(prediction)
    actual_digit = np.argmax(y_test[index])
    one_hot_vector = y_test[index]

    axes[i].imshow(
        image.reshape(28, 28),
        cmap="gray"
    )

    axes[i].set_title(
        f"Model 2 - Example {i + 1}",
        fontsize=12
    )

    axes[i].axis("off")

    result_text = (
        f"Actual Digit   : {actual_digit}\n"
        f"One-Hot Vector : {one_hot_vector.astype(int)}\n"
        f"Predicted Digit: {predicted_digit}"
    )

    axes[i].text(
        0.5,
        -0.25,
        result_text,
        transform=axes[i].transAxes,
        ha="center",
        va="top",
        fontsize=10
    )

plt.tight_layout()
plt.show()




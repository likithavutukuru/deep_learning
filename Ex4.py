import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

train = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=1,
    image_size=(224,224),
    batch_size=8,
    shuffle=False
)

test = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=1,
    image_size=(224,224),
    batch_size=8,
    shuffle=False
)

classes = train.class_names

cnn = VGG16(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

train_images = np.concatenate([x.numpy() for x,y in train])
train_labels = np.concatenate([y.numpy() for x,y in train])

test_images = np.concatenate([x.numpy() for x,y in test])
test_labels = np.concatenate([y.numpy() for x,y in test])

train_features = cnn.predict(preprocess_input(train_images))
test_features = cnn.predict(preprocess_input(test_images))

print("CNN Feature Vector:", train_features.shape)

X = train_features.reshape(
    train_features.shape[0],
    1,
    train_features.shape[1]
)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, train_labels, epochs=10)

test_X = test_features.reshape(
    test_features.shape[0],
    1,
    test_features.shape[1]
)

pred = model.predict(test_X)

for i in range(min(5, len(pred))):

    predicted_class = classes[np.argmax(pred[i])]

    expected = "A photo of a " + classes[test_labels[i]] + "."
    generated = "A photo of a " + predicted_class + "."

    print("Expected :", expected)
    print("Generated:", generated)
    print()
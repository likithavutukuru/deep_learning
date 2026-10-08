import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

train = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=1,
    image_size=(224,224),
    batch_size=8
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

base = VGG16(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224,224,3)
)

base.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Lambda(preprocess_input),
    base,
    tf.keras.layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, epochs=3)

pred = model.predict(test)

y_pred = np.argmax(pred, axis=1)

y_true = np.concatenate([y.numpy() for x,y in test])

print("Feature Vector Dimension:", base.output_shape[-1])

print("Accuracy:", accuracy_score(y_true, y_pred))

print(classification_report(
    y_true,
    y_pred,
    target_names=classes
))

cm = confusion_matrix(y_true, y_pred)

ConfusionMatrixDisplay(
    cm,
    display_labels=classes
).plot()

plt.show()
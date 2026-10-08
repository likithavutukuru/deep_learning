import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

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

model = VGG16(weights="imagenet")

images, labels = next(iter(test))

x = preprocess_input(images.numpy().copy())

predictions = model.predict(x)

for i in range(min(5, len(images))):

    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title("Original: " + classes[labels[i]])
    plt.axis("off")
    plt.show()

    print("Original Label:", classes[labels[i]])

    top5 = decode_predictions(predictions[i:i+1], top=5)[0]

    print("Top 5 Predictions")

    for _, name, score in top5:
        print(name, round(score * 100, 2), "%")

    print()

print("Original shape:", images.shape)
print("Preprocessed shape:", x.shape)
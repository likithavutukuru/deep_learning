import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(224,224),
    batch_size=1
)

images, labels = next(iter(data))

model = VGG16(weights="imagenet", include_top=False)

layer_names = [
    "block1_conv1",
    "block3_conv1",
    "block5_conv1"
]

outputs = []

for name in layer_names:
    layer = model.get_layer(name)
    outputs.append(layer.output)

feature_model = Model(model.input, outputs)

x = preprocess_input(images.numpy().copy())

feature_maps = feature_model.predict(x)

plt.imshow(images[0].numpy().astype("uint8"))
plt.title("Input Image")
plt.axis("off")
plt.show()

for name, feature in zip(layer_names, feature_maps):

    print(name, feature.shape)

    plt.figure(figsize=(8,4))

    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.imshow(feature[0,:,:,i], cmap="gray")
        plt.axis("off")

    plt.suptitle(name)
    plt.show()
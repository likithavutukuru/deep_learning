import tensorflow as tf
import matplotlib.pyplot as plt

SIZE = 64
NOISE = 100

data = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(SIZE,SIZE),
    batch_size=32
)

classes = data.class_names
n = len(classes)

data = data.map(
    lambda x,y: ((tf.cast(x, tf.float32)-127.5)/127.5, y)
)

generator = tf.keras.Sequential([
    tf.keras.layers.Input((NOISE + n,)),

    tf.keras.layers.Dense(8*8*128),

    tf.keras.layers.Reshape((8,8,128)),

    tf.keras.layers.Conv2DTranspose(
        64, 4, 2, padding="same", activation="relu"
    ),

    tf.keras.layers.Conv2DTranspose(
        32, 4, 2, padding="same", activation="relu"
    ),

    tf.keras.layers.Conv2DTranspose(
        3, 4, 2, padding="same", activation="tanh"
    )
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Input((SIZE,SIZE,3+n)),

    tf.keras.layers.Conv2D(
        64, 4, 2, padding="same"
    ),

    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Conv2D(
        128, 4, 2, padding="same"
    ),

    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1)
])

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

g_optimizer = tf.keras.optimizers.Adam(0.0002)
d_optimizer = tf.keras.optimizers.Adam(0.0002)

def train(images, labels):

    batch = tf.shape(images)[0]

    text = tf.one_hot(labels, n)

    noise = tf.random.normal((batch, NOISE))

    g_input = tf.concat([noise, text], axis=1)

    condition = tf.reshape(text, (batch,1,1,n))
    condition = tf.tile(condition, (1,SIZE,SIZE,1))

    with tf.GradientTape() as gt, tf.GradientTape() as dt:

        fake = generator(g_input)

        real_input = tf.concat([images, condition], axis=-1)
        fake_input = tf.concat([fake, condition], axis=-1)

        real_output = discriminator(real_input)
        fake_output = discriminator(fake_input)

        g_loss = loss(
            tf.ones_like(fake_output),
            fake_output
        )

        d_loss = (
            loss(tf.ones_like(real_output), real_output)
            +
            loss(tf.zeros_like(fake_output), fake_output)
        )

    g_grad = gt.gradient(
        g_loss,
        generator.trainable_variables
    )

    d_grad = dt.gradient(
        d_loss,
        discriminator.trainable_variables
    )

    g_optimizer.apply_gradients(
        zip(g_grad, generator.trainable_variables)
    )

    d_optimizer.apply_gradients(
        zip(d_grad, discriminator.trainable_variables)
    )

for epoch in range(5):

    for images, labels in data:
        train(images, labels)

    print("Epoch", epoch+1)

class_id = 0

text = tf.one_hot([class_id], n)

noise = tf.random.normal((1, NOISE))

generated = generator(
    tf.concat([noise, text], axis=1),
    training=False
)

image = (generated[0] + 1) / 2

print("Text: A photo of a", classes[class_id])

plt.imshow(image)
plt.title("Generated " + classes[class_id])
plt.axis("off")
plt.show()
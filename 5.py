import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load dataset
ds = tf.keras.utils.image_dataset_from_directory(
    "ImageNet-Mini/images", image_size=(32,32),
    batch_size=32, seed=42
)
names = ds.class_names
captions = tf.constant(["a photo of a "+n.replace("_"," ") for n in names])

# 2. Text encoder
enc = tf.keras.layers.TextVectorization(output_mode="multi_hot")
enc.adapt(captions)
T = enc.vocabulary_size()

# 3. Generator
z, t = tf.keras.Input((100,)), tf.keras.Input((T,))
x = tf.keras.layers.Concatenate()([z,t])
x = tf.keras.layers.Dense(8*8*128, activation="relu")(x)
x = tf.keras.layers.Reshape((8,8,128))(x)
x = tf.keras.layers.UpSampling2D()(x)
x = tf.keras.layers.Conv2D(64,3,padding="same",activation="relu")(x)
x = tf.keras.layers.UpSampling2D()(x)
out = tf.keras.layers.Conv2D(3,3,padding="same",activation="tanh")(x)
G = tf.keras.Model([z,t],out)

# 4. Discriminator
i, c = tf.keras.Input((32,32,3)), tf.keras.Input((T,))
x = tf.keras.layers.Flatten()(i)
x = tf.keras.layers.Concatenate()([x,c])
x = tf.keras.layers.Dense(128,activation="relu")(x)
D = tf.keras.Model([i,c],tf.keras.layers.Dense(1)(x))

# 5. Loss and optimizers
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
go, do = tf.keras.optimizers.Adam(0.0002), tf.keras.optimizers.Adam(0.0002)

# 6. Train GAN
for epoch in range(5):
    for real, labels in ds:
        real = tf.cast(real,tf.float32)/127.5-1
        text = enc(tf.gather(captions,labels))
        noise = tf.random.normal((tf.shape(real)[0],100))

        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = G([noise,text],training=True)
            r, f = D([real,text]), D([fake,text])
            gl = loss(tf.ones_like(f),f)
            dl = (loss(tf.ones_like(r),r)+loss(tf.zeros_like(f),f))/2

        go.apply_gradients(zip(gt.gradient(gl,G.trainable_variables),G.trainable_variables))
        do.apply_gradients(zip(dt.gradient(dl,D.trainable_variables),D.trainable_variables))

    print("Epoch",epoch+1,"G:",float(gl),"D:",float(dl))

# 7. Generate image matching a real image's class
real, labels = next(iter(ds))
idx = int(labels[0].numpy())
description = captions[idx].numpy().decode()
text = enc([description])
image = G([tf.random.normal((1,100)),text],training=False)[0].numpy()

# 8. Compare images
plt.figure(figsize=(7,3))
plt.subplot(1,2,1)
plt.imshow(real[0].numpy().astype("uint8"))
plt.title("Real: "+names[idx])
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(np.clip((image+1)/2,0,1))
plt.title("Generated: "+description)
plt.axis("off")
plt.show()

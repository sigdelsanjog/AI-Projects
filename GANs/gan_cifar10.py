import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import threading
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# Define the generator model
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(32 * 32 * 3, activation='tanh'))
    model.add(Reshape((32, 32, 3)))
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build and compile GAN models
def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    discriminator.trainable = False
    z = Input(shape=(100,))
    img = generator(z)
    validity = discriminator(img)
    gan = Model(z, validity)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return gan

# Function for live plot of losses
def live_plot(losses):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    d_losses, g_losses = losses['D'], losses['G']
    line1, = ax.plot(d_losses, label="Discriminator Loss")
    line2, = ax.plot(g_losses, label="Generator Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    
    while True:
        line1.set_ydata(d_losses)
        line2.set_ydata(g_losses)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)

# Function to save sample images during training
def sample_images(epoch, generator, image_grid_rows=4, image_grid_columns=4):
    noise = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images 0 - 1
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4))
    count = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[count])
            axs[i, j].axis('off')
            count += 1
    plt.show()

# Function to train GAN
def train_gan(generator, discriminator, gan, epochs, batch_size=32, sample_interval=100):
    # Load and preprocess data
    (X_train, _), (_, _) = cifar10.load_data()
    X_train = (X_train - 127.5) / 127.5  # Normalize to [-1, 1]
    half_batch = int(batch_size / 2)
    
    # Dictionary to store losses for live plotting
    losses = {'D': [], 'G': []}
    
    # Start the live plotting in a separate thread
    plot_thread = threading.Thread(target=live_plot, args=(losses,))
    plot_thread.start()
    
    for epoch in range(epochs):
        # Select a random half batch of real images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[idx]
        
        # Generate fake images
        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_imgs = generator.predict(noise)
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, valid_y)
        
        # Store losses for live plot
        losses['D'].append(d_loss[0])
        losses['G'].append(g_loss)
        
        # Print progress
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")
            sample_images(epoch, generator)

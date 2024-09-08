import os
import glob
import librosa
import numpy as np
from tensorflow import keras

def preprocess_audio():
    global spectrograms
    spectrograms = []
    tracks_folder = 'tracks'
    mp3_files = glob.glob(os.path.join(tracks_folder, '*.mp3'))
    for mp3_file in mp3_files:
        y, sr = librosa.load(mp3_file)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Resize all spectrograms to the same shape
        spectrogram = librosa.util.fix_length(spectrogram, size=128, axis=-1)
        spectrograms.append(spectrogram)

    return np.array(spectrograms)

def build_generator():
    model = keras.Sequential([
        keras.layers.Dense(256, input_shape=(100,)),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(512),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(1024),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.BatchNormalization(momentum=0.8),
        keras.layers.Dense(np.prod(spectrogram_shape), activation='tanh'),
        keras.layers.Reshape(spectrogram_shape)
    ])
    return model

def build_discriminator():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=spectrogram_shape),
        keras.layers.Dense(512),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dense(256),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_gan(generator, discriminator, dataset, epochs=1000, batch_size=32):
    for epoch in range(epochs):
        real_samples = dataset[np.random.randint(0, dataset.shape[0], batch_size)]
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_samples = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_samples, np.zeros((batch_size, 1)))
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
        
    gan.save('gan_model.h5')



spectrogram_shape = preprocess_audio()[0].shape

# Build and compile GAN
generator = build_generator()
generator.save('generator_model.h5')

discriminator = build_discriminator()
discriminator.save('discriminator_model.h5')
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))

gan_input = keras.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))

# Train GAN
train_gan(generator, discriminator, np.array(spectrograms))

import numpy as np
from tensorflow import keras


def load_and_train_gan(dataset, additional_epochs=1000, batch_size=32):
    # Load the pre-trained models
    generator = keras.models.load_model('generator_model.h5')
    discriminator = keras.models.load_model('discriminator_model.h5')
    gan = keras.models.load_model('gan_model.h5')
    
    # Further train the models
    for epoch in range(additional_epochs):
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
    
    # Save the further trained models
    generator.save('generator_model.h5')
    discriminator.save('discriminator_model.h5')
    gan.save('gan_model.h5')


dataset = ...
load_and_train_gan(dataset)
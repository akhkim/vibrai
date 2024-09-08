import librosa
import numpy as np
import soundfile as sf
from tensorflow import keras


def spectrogram_to_audio(spectrogram, sr=22050, n_iter=10):
    spec_linear = librosa.db_to_power(spectrogram)
    audio_signal = librosa.griffinlim(spec_linear, n_iter=n_iter)
    
    return audio_signal

def save_audio(audio_signal, filename, sr=22050):
    sf.write(filename, audio_signal, sr)


generator = keras.models.load_model('generator_model.h5')
discriminator = keras.models.load_model('discriminator_model.h5')
gan = keras.models.load_model('gan_model.h5')

# Generate new music
noise = np.random.normal(0, 1, (1, 100))
generated_spectrogram = generator.predict(noise)

audio = spectrogram_to_audio(generated_spectrogram[0])
save_audio(audio, 'generated_audio.wav')
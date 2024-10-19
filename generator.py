import os
import torch
import soundfile as sf
from train import MaskedGenerator, mel_spectrogram_to_music

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_generator(model_path, latent_dim=512, max_length=512):
    generator = MaskedGenerator(latent_dim=latent_dim, max_length=max_length).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    return generator

def generate_mel_spectrogram(generator, latent_dim=512, max_length=512):
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, device=device)
        length = torch.tensor([max_length], device=device)
        mel_spec = generator(noise, length)
    return mel_spec.squeeze().cpu()

def save_audio(y, sr, filename="generated_music.wav"):
    sf.write(filename, y, sr)
    print(f"Audio saved as {filename}")

def main():
    model_path = os.path.join('saved_models', 'generator.pth')
    generator = load_generator(model_path)
    mel_spec = generate_mel_spectrogram(generator)
    audio, sr = mel_spectrogram_to_music(mel_spec)
    save_audio(audio, sr, "generated_music.wav")

if __name__ == "__main__":
    main()
    print("Music generation complete!")
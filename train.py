import os
import glob
import librosa
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import soundfile as sf
from torchvision.models import vgg16, VGG16_Weights
from torchaudio.transforms import MelSpectrogram, Resample


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_audio(max_length=512):
    tracks = []
    specs = []
    lengths = []
    mel_spectrogram = MelSpectrogram(
        sample_rate=22050,
        n_fft=2048, 
        hop_length=512,
        n_mels=128
    )
    
    for file in glob.glob(os.path.join("tracks", '*.mp3' or '*.wav')):
        tracks.append(file)

    for track in tracks:
        waveform, sample_rate = torchaudio.load(track)
        if sample_rate != 22050:
            resampler = Resample(sample_rate, 22050)
            waveform = resampler(waveform)
        
        # Ensure the waveform is mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        mel_spec = mel_spectrogram(waveform)
        
        # Store the original length
        original_length = mel_spec.shape[2]
        lengths.append(original_length)
        
        # Pad the spectrogram to the max_length
        if original_length < max_length:
            pad_length = max_length - original_length
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_length))
        elif original_length > max_length:
            mel_spec = mel_spec[:, :, :max_length]
        
        specs.append(mel_spec.squeeze().unsqueeze(0))

    return torch.stack(specs), torch.tensor(lengths)

class MaskedGenerator(nn.Module):
    def __init__(self, latent_dim=512, max_length=512):
        super(MaskedGenerator, self).__init__()
        self.max_length = max_length
        self.latent_dim = latent_dim
        
        self.model = nn.Sequential(
            # Initial dense layer
            nn.Linear(latent_dim, 256 * 8 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 8, 16)),

            # First transposed convolution
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Second transposed convolution
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Third transposed convolution
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Final transposed convolution
            nn.ConvTranspose2d(32, 1, kernel_size=(4, 8), stride=(2, 4), padding=(1, 2)),
            nn.Tanh()
        )
    
    def forward(self, x, lengths):
        x = x.view(x.size(0), self.latent_dim)
        output = self.model(x)
        # Apply mask based on lengths
        mask = torch.arange(self.max_length).expand(output.size(0), -1) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(2).float().to(output.device)
        return output * mask

class MaskedDiscriminator(nn.Module):
    def __init__(self):
        super(MaskedDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x, lengths):
        mask = torch.arange(x.size(3)).expand(x.size(0), -1) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(2).float().to(x.device)
        masked_x = x * mask
        return self.model(masked_x)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3

def compute_gradient_penalty(discriminator, real_samples, fake_samples, lengths):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, lengths)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_model(generator, discriminator, specs, lengths, num_epochs=5000, batch_size=64, lr=0.001, beta1=0.5, beta2=0.9):
    perceptual_loss = PerceptualLoss().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    dataset = torch.utils.data.TensorDataset(specs, lengths)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for i, (real_specs, real_lengths) in enumerate(dataloader):
            batch_size = real_specs.size(0)
            real_specs = real_specs.to(device)
            real_lengths = real_lengths.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            
            noise = torch.randn(batch_size, generator.latent_dim, device=device)
            fake_specs = generator(noise, real_lengths)
            
            real_validity = discriminator(real_specs, real_lengths)
            fake_validity = discriminator(fake_specs.detach(), real_lengths)
            
            gradient_penalty = compute_gradient_penalty(discriminator, real_specs, fake_specs.detach(), real_lengths)
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            if i % 5 == 0:
                optimizer_G.zero_grad()
                
                gen_specs = generator(noise, real_lengths)
                fake_validity = discriminator(gen_specs, real_lengths)
                
                # Compute perceptual loss
                real_features = perceptual_loss(real_specs.repeat(1, 3, 1, 1))
                fake_features = perceptual_loss(gen_specs.repeat(1, 3, 1, 1))
                p_loss = sum(torch.mean((real - fake) ** 2) for real, fake in zip(real_features, fake_features))
                
                g_loss = -torch.mean(fake_validity) + 0.1 * p_loss
                g_loss.backward()
                optimizer_G.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    return generator, discriminator

def save_models(generator, discriminator, save_dir='saved_models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(generator.state_dict(), os.path.join(save_dir, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))
    print(f"Models saved in {save_dir}")

def mel_spectrogram_to_music(mel_spec, sr=22050):
    mel_spec = mel_spec.numpy()
    mel_spec = librosa.db_to_power(mel_spec)

    y = librosa.feature.inverse.mel_to_audio(
        mel_spec, 
        sr=sr, 
        n_iter=10,
        hop_length=512
    )

    return y, sr

def save_audio(y, sr, filename="generated_music.wav"):
    sf.write(filename, y, sr)
    print(f"Audio saved as {filename}")

if __name__ == "__main__":
    specs, lengths = prepare_audio()
    generator = MaskedGenerator(latent_dim=512, max_length=specs.size(3)).to(device)
    discriminator = MaskedDiscriminator().to(device)
    
    # Train the models
    generator, discriminator = train_model(generator, discriminator, specs, lengths)
    
    # Save the trained models
    save_models(generator, discriminator)
    
    # Generate a sample
    with torch.no_grad():
        sample_noise = torch.randn(1, generator.latent_dim, device=device)
        sample_length = torch.tensor([specs.size(3)], device=device)
        sample_spec = generator(sample_noise, sample_length)
    
    audio, sr = mel_spectrogram_to_music(sample_spec.squeeze().cpu())    
    save_audio(audio, sr)

    print("Training and generation complete!")
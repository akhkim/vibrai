import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train import MaskedGenerator, MaskedDiscriminator, prepare_audio, PerceptualLoss, compute_gradient_penalty, mel_spectrogram_to_music, save_audio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(model_dir='saved_models'):
    generator = MaskedGenerator(latent_dim=512, max_length=512).to(device)
    discriminator = MaskedDiscriminator().to(device)
    
    generator.load_state_dict(torch.load(os.path.join(model_dir, 'generator.pth')))
    discriminator.load_state_dict(torch.load(os.path.join(model_dir, 'discriminator.pth')))
    
    return generator, discriminator

def fine_tune(generator, discriminator, specs, lengths, num_epochs=1000, batch_size=32, lr=0.0001):
    perceptual_loss = PerceptualLoss().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    dataset = TensorDataset(specs, lengths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

def generate_sample(generator, max_length=512):
    with torch.no_grad():
        sample_noise = torch.randn(1, generator.latent_dim, device=device)
        sample_length = torch.tensor([max_length], device=device)
        sample_spec = generator(sample_noise, sample_length)
    
    audio, sr = mel_spectrogram_to_music(sample_spec.squeeze().cpu())
    return audio, sr


if __name__ == "__main__":
    # Load pre-trained models
    generator, discriminator = load_models()
    
    # Prepare new data for fine-tuning
    new_specs, new_lengths = prepare_audio()
    
    # Fine-tune the models
    generator, discriminator = fine_tune(generator, discriminator, new_specs, new_lengths)
    
    # Save fine-tuned models
    torch.save(generator.state_dict(), 'saved_models/fine_tuned_generator.pth')
    torch.save(discriminator.state_dict(), 'saved_models/fine_tuned_discriminator.pth')
    
    # Generate a sample using the fine-tuned model
    audio, sr = generate_sample(generator)
    save_audio(audio, sr, "fine_tuned_sample.wav")
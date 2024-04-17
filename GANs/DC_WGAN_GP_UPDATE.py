import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime

class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_dim=3, channels=3):
        super(Generator, self).__init__()
        self.initial_size = 8  # This should be calculated based on the desired output structure
        self.initial_channels = 512  # Start with a high number of channels and decrease

        # Linear layer to expand noise vector
        self.l1 = nn.Sequential(
            nn.Linear(noise_dim + feature_dim, self.initial_channels * self.initial_size)
        )

        # Convolution blocks to upscale and shape into the correct output format
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(self.initial_channels),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(self.initial_channels, 256, 2, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2,True),
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(64, channels, 4, stride=2, padding=5),
            nn.Tanh()  # Normalizing output to [-1, 1]
        )

    def forward(self, noise, features):
        # Combine noise and features and pass through the network
        x = torch.cat((noise, features), dim=1)
        x = self.l1(x)
        x = x.view(x.size(0), self.initial_channels, self.initial_size)  # Reshape to match conv layer
        img = self.conv_blocks(x)
        return img


class Discriminator(nn.Module):
    def __init__(self, feature_dim=3, channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channels + feature_dim, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )
        
        # Calculate flatten size 
        with torch.no_grad():  # Ensuring no gradient is computed
            dummy_input = torch.zeros(1, channels + feature_dim, 1000) 
            dummy_output = self.model(dummy_input)
            self.flatten_size = int(np.prod(dummy_output.size()[1:]))

        self.final_linear = nn.Linear(self.flatten_size, 1)

    def forward(self, img, features):
        features = features.unsqueeze(2).expand(-1, -1, img.size(2))
        x = torch.cat((img, features), 1)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.final_linear(x)

class DC_WGAN_GP:
    def __init__(self, channels=3, batch_size=50, noise_dim=100, feature_dim=3):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print('Using: ', {self.device})
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.lambda_gp = 0.1

        self.generator = Generator(noise_dim=noise_dim, feature_dim=feature_dim, channels=channels).to(self.device)
        self.discriminator = Discriminator(feature_dim=feature_dim, channels=channels).to(self.device)
        
        self.optimiser_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.2, 0.999))
        self.optimiser_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.9, 0.999))
        

    def compute_gradient_penalty(self, real_samples, fake_samples, features):
        alpha = torch.rand(real_samples.size(0), 1, 1, device=self.device)
        alpha = alpha.expand_as(real_samples)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        d_interpolates = self.discriminator(interpolated, features)
        fake = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolated,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient_penalty = self.lambda_gp * ((gradients.norm(2, dim=(1, 2)) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, data_loader, epochs, save_interval=100):
        self.d_losses_real, self.d_losses_fake, self.g_losses = [], [], []
        current_datetime = datetime.datetime.now().strftime('%d-%m_%H-%M')

        save_path = f'./dc_wgan_gp_generated_samples_{current_datetime}'
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(epochs):
            for i, (imgs, features) in enumerate(data_loader):
                imgs = imgs.view(-1, 3, 1000).to(self.device)
                features = features.float().to(self.device)

                # Train Discriminator
                self.optimiser_D.zero_grad()
                noise = torch.randn(imgs.size(0), self.noise_dim, device=self.device)
                fake_imgs = self.generator(noise, features).detach()
                real_loss = self.discriminator(imgs, features).mean()
                fake_loss = self.discriminator(fake_imgs, features).mean()
                gradient_penalty = self.compute_gradient_penalty(imgs.data, fake_imgs.data, features)
                d_loss = fake_loss - real_loss + gradient_penalty
                d_loss.backward()
                self.optimiser_D.step()

                # Train Generator every nth step of discriminator
                if i % 5 == 0:
                    self.optimiser_G.zero_grad()
                    gen_imgs = self.generator(noise, features)
                    g_loss = -self.discriminator(gen_imgs, features).mean()
                    g_loss.backward()
                    self.optimiser_G.step()

            # Logging
            if epoch % save_interval == 0:
                with torch.no_grad():
                    sample_noise = torch.randn(self.batch_size, self.noise_dim, device=self.device)
                    sample_features = torch.randn(self.batch_size, self.feature_dim, device=self.device)
                    generated_samples = self.generator(sample_noise, sample_features)
                    np.save(os.path.join(save_path, f'generated_samples_epoch_{epoch}_batch_{i}.npy'), generated_samples.cpu().numpy())

                self.d_losses_real.append(real_loss.item())
                self.d_losses_fake.append(fake_loss.item())
                self.g_losses.append(g_loss.item())

            print(f"Epoch: {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

        # Save the final models
        torch.save(self.generator.state_dict(), f'./model_states/dc_wgan_generator_{current_datetime}.pth')
        torch.save(self.discriminator.state_dict(),f'./model_states/dc_wgan_discriminator_{current_datetime}.pth')

        # Plotting the training losses
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.d_losses_real, label="Discriminator Real Loss")
        plt.plot(self.d_losses_fake, label="Discriminator Fake Loss")
        plt.plot(self.g_losses, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()

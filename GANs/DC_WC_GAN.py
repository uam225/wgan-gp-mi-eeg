import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_dim=3, channels=3):
        super(Generator, self).__init__()
        self.init_size = 8

        self.l1 = nn.Sequential(nn.Linear(noise_dim + feature_dim, 128 * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(64, channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, features):
        combined_input = torch.cat((noise, features), dim=1)
        gen_input = self.l1(combined_input)
        gen_input = gen_input.view(-1, 128, self.init_size)
        signal = self.conv_blocks(gen_input)
        return signal

class Critic(nn.Module):
    def __init__(self, feature_dim=3, channels=3):
        super(Critic, self).__init__()
        self.channels = channels
        self.feature_dim = feature_dim
        self.conv_model = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )
        self.fc_features = nn.Linear(feature_dim, 512)  # Feature dimension to match conv output
        self.fc_final = nn.Linear(64 * 512 , 1)  # Combining conv output and feature processing output

    def forward(self, img, features):
        img_output = self.conv_model(img).view(img.size(0), -1)
        print("Conv output shape: ", img_output.shape)
        features_output = self.fc_features(features)
        combined = torch.cat((img_output, features_output), dim=1)
        decision = self.fc_final(combined)
        return decision

class WCDCGAN:
    def __init__(self, channels=3, batchsize=50, noise_dim=100, feature_dim=3, clip_value=0.01):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.channels = channels
        self.batchsize = batchsize
        self.noise_dim = noise_dim  
        self.feature_dim = feature_dim
        self.clip_value = clip_value  

        self.generator = Generator(noise_dim=noise_dim, feature_dim=feature_dim, channels=channels).to(self.device)
        self.critic = Critic(feature_dim=feature_dim, channels=channels).to(self.device) 

        self.optimiser_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimiser_D = optim.Adam(self.critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, data_loader, epochs):
        d_losses, g_losses = [], []
        d_real_losses, d_fake_losses = [], []

        for epoch in range(epochs):
            d_loss_sum, d_real_loss_sum, d_fake_loss_sum = 0.0, 0.0, 0.0
            n_batches = 0

            for i, (imgs, features) in enumerate(data_loader):
                real_imgs = imgs.float().to(self.device)
                real_imgs = real_imgs.view(-1, 3, 1000)
                print("Real img shape: ", real_imgs.shape)
                features = features.float().to(self.device)

                noise = torch.randn(imgs.size(0), self.noise_dim, device=self.device)
                fake_imgs = self.generator(noise, features)

                self.optimiser_D.zero_grad()
                real_decision = self.critic(real_imgs, features)
                real_loss = real_decision.mean()
                real_loss.backward(retain_graph=True)
                
                fake_decision = self.critic(fake_imgs.detach(), features)
                fake_loss = fake_decision.mean()
                fake_loss.backward(retain_graph=True)
                d_loss = real_loss - fake_loss

                self.optimiser_D.step()

                if self.clip_value:
                    for p in self.critic.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)

                if i % 10 == 0:
                    self.optimiser_G.zero_grad()
                    g_loss = -self.critic(fake_imgs, features).mean()
                    g_loss.backward()
                    self.optimiser_G.step()
                    g_losses.append(g_loss.item())

                d_loss_sum += d_loss.item()
                d_real_loss_sum += real_loss.item()
                d_fake_loss_sum += fake_loss.item()
                n_batches += 1

            d_losses.append(d_loss_sum / n_batches)
            d_real_losses.append(d_real_loss_sum / n_batches)
            d_fake_losses.append(d_fake_loss_sum / n_batches)

            print(f"Epoch: {epoch}, D Loss: {d_loss_sum / n_batches}, G Loss: {np.mean(g_losses[-n_batches:])}")

        plt.figure(figsize=(15, 5))
        plt.plot(d_real_losses, label='Critic Loss on Real')
        plt.plot(d_fake_losses, label='Critic Loss on Fake')
        plt.plot(g_losses, label='Generator Loss')
        plt.title('Combined Losses Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_plots/critic_losses.png')
        plt.show()

        return fake_imgs.detach()

# Example of how you might initialize and train the model:
# wcdcgan = WCDCGAN(channels=3, batchsize=50, noise_dim=100, feature_dim=3)
# data_loader = DataLoader(...)  # Assume you have a DataLoader for your data
# wcdcgan.train(data_loader, epochs=10)

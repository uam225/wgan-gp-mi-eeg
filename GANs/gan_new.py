import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Assuming that EEGFeatureDataset is defined in another script
from data_loader import EEGFeatureDataset

class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_dim=3000):  # Adjust output_dim to match combined EEG and feature dimensions
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.02, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noise):
        return self.model(noise)


class Discriminator(nn.Module):
    def __init__(self, input_dim):  # input_dim is now the combined EEG and CSP features dimension
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, combined_data):
        return self.model(combined_data)


class GAN:
    def __init__(self, combined_dim, batchsize=50, noise_dim=100):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.batchsize = batchsize
        self.noise_dim = noise_dim

        self.generator = Generator(noise_dim=noise_dim, output_dim=combined_dim).to(self.device)
        self.discriminator = Discriminator(input_dim=combined_dim).to(self.device)

        self.optimiser_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimiser_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.loss = nn.BCELoss()

    def train(self, data_loader, epochs):
        d_losses, g_losses = [], []
        d_real_acc, d_fake_acc = [], []

        for epoch in range(epochs):
            d_loss_sum, g_loss_sum = 0.0, 0.0
            real_acc_sum, fake_acc_sum = 0.0, 0.0
            n_batches = 0

            for i, combined_data in enumerate(data_loader):
                batch_size = combined_data.size(0)
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                # Train Discriminator
                self.optimiser_D.zero_grad()
                real_data = combined_data.to(self.device)
                decision_real = self.discriminator(real_data)
                d_real_loss = self.loss(decision_real, real_labels)

                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_data = self.generator(noise)
                decision_fake = self.discriminator(fake_data.detach())
                d_fake_loss = self.loss(decision_fake, fake_labels)

                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optimiser_D.step()

                # Train Generator
                self.optimiser_G.zero_grad()
                decision_fake = self.discriminator(fake_data)
                g_loss = self.loss(decision_fake, real_labels)
                g_loss.backward()
                self.optimiser_G.step()

                # Track progress
                d_loss_sum += d_loss.item()
                g_loss_sum += g_loss.item()
                real_acc_sum += torch.mean((decision_real >= 0.5).float()).item()
                fake_acc_sum += torch.mean((decision_fake < 0.5).float()).item()
                n_batches += 1

            # Average the loss and accuracy over all batches
            d_losses.append(d_loss_sum / n_batches)
            g_losses.append(g_loss_sum / n_batches)
            d_real_acc.append(real_acc_sum / n_batches)
            d_fake_acc.append(fake_acc_sum / n_batches)

            print(f"Epoch {epoch}: D Loss: {d_losses[-1]}, G Loss: {g_losses[-1]}, Real Acc: {d_real_acc[-1]}, Fake Acc: {d_fake_acc[-1]}")

        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Plot accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(d_real_acc, label='Discriminator Real Accuracy')
        plt.plot(d_fake_acc, label='Discriminator Fake Accuracy')
        plt.title("Accuracies")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
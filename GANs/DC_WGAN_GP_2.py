import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_dim=3, channels=3):
        super(Generator, self).__init__()
        self.init_size = 10  # should correspond to the actual spatial size after processing
        self.l1 = nn.Sequential(nn.Linear(noise_dim + feature_dim, 128 * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=4),
            nn.Conv1d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, features):
        x = torch.cat((noise, features), dim=1)
        x = self.l1(x)
        x = x.view(x.shape[0], 128, -1)  # dynamically compute the size instead of fixed
        img = self.conv_blocks(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, feature_dim=3, channels=3):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(channels + feature_dim, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.5, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Flatten(),
        )
        self.linear = nn.Linear(512 * 16, 1)  # Assuming fixed size after convolutions

    def forward(self, img, features):
        img_features = torch.cat((img, features), 1)
        conv_out = self.conv_blocks(img_features)
        return self.linear(conv_out)

class WGAN_GP:
    def __init__(self, channels=3, batch_size=50, noise_dim=100, feature_dim=3):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.channels = channels
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim

        self.generator = Generator(noise_dim, feature_dim, channels).to(self.device)
        self.discriminator = Discriminator(feature_dim, channels).to(self.device)

        self.optimiser_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimiser_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.d_losses_real, self.d_losses_fake, self.g_losses = [], [], []
        self.d_accuracies_real, self.d_accuracies_fake = [], []

    def compute_gradient_penalty(self, real_samples, fake_samples, features):
        alpha = torch.rand((real_samples.size(0), 1, 1), device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, features)
        fake = torch.ones(d_interpolates.size(), device=self.device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            for i, (imgs, features) in enumerate(data_loader):
                imgs = imgs.to(self.device)
                features = features.to(self.device).float()
                real_imgs = imgs.float()

                # Train Discriminator
                self.optimiser_D.zero_grad()
                noise = torch.randn(self.batch_size, self.noise_dim, device=self.device)
                fake_imgs = self.generator(noise, features).detach()
                real_loss = self.discriminator(real_imgs, features).mean()
                fake_loss = self.discriminator(fake_imgs, features).mean()
                gradient_penalty = self.compute_gradient_penalty(real_imgs, fake_imgs, features)
                d_loss = fake_loss - real_loss + gradient_penalty
                d_loss.backward()
                self.optimiser_D.step()

                self.d_losses_real.append(real_loss.item())
                self.d_losses_fake.append(fake_loss.item())

                d_accuracy_real = ((self.discriminator(real_imgs, features) > 0).float().mean()).item()
                d_accuracy_fake = ((self.discriminator(fake_imgs, features) < 0).float().mean()).item()
                self.d_accuracies_real.append(d_accuracy_real)
                self.d_accuracies_fake.append(d_accuracy_fake)

                # Train Generator
                if i % 5 == 0:
                    self.optimiser_G.zero_grad()
                    gen_imgs = self.generator(noise, features)
                    g_loss = -self.discriminator(gen_imgs, features).mean()
                    g_loss.backward()
                    self.optimiser_G.step()
                    self.g_losses.append(g_loss.item())

                print(f"Epoch: {epoch}, Batch: {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    # Plotting the losses and accuracies
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.d_losses_real, label="Discriminator Real Loss")
        plt.plot(self.d_losses_fake, label="Discriminator Fake Loss")
        plt.plot(self.g_losses, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.d_accuracies_real, label="Discriminator Real Accuracy")
        plt.plot(self.d_accuracies_fake, label="Discriminator Fake Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

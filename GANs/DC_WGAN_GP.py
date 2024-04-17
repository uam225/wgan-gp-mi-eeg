import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
import datetime

class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_dim=3, channels=3):
        super(Generator, self).__init__()
        self.init_size = 10  
        self.l1 = nn.Sequential(nn.Linear(noise_dim + feature_dim, 128 * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=4),
            nn.Conv1d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, features):
        x = torch.cat((noise, features), dim=1)
        x = self.l1(x)
        x = x.view(x.shape[0], 128, self.init_size)
        img = self.conv_blocks(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, feature_dim=3, channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channels+feature_dim, 64, 3, stride=2, padding=1),
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
            nn.Dropout(0.5),
            nn.Conv1d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )

        self.flatten_size = self.compute_flatten_size()
        self.linear_layer = nn.Linear(1024 * self.flatten_size, 1)

    def compute_flatten_size(self):
        #compute the size needed for the linear layer
        return 1000 // (2 ** 6)#16  

    def forward(self, img, features):
        features = features.unsqueeze(2)
        features = features.expand(img.size(0), features.size(1), img.size(2))
        x = torch.cat((img, features), 1)
        return self.model(x)

class WGAN_GP:
    def __init__(self, channels=3, batch_size=50, noise_dim=100, feature_dim=3):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.channels = channels
        self.batchsize = batch_size
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.lambda_gp = 0.1 #gp hyperparam
        

        self.generator = Generator(noise_dim=noise_dim, feature_dim=feature_dim, channels=channels).to(self.device)
        self.discriminator = Discriminator(feature_dim=feature_dim).to(self.device)

        self.optimiser_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.2, 0.999))
        self.optimiser_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.9, 0.999))

    def compute_gradient_penalty(self, real_samples, fake_samples, features):
        alpha = torch.rand((real_samples.size(0), 1, 1), device=self.device)
        fake_samples_interpolated = F.interpolate(fake_samples, size=real_samples.size()[2])
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples_interpolated)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates, features)
        fake = Variable(torch.ones(d_interpolates.size(), device=self.device), requires_grad=False)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = self.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, data_loader, epochs, save_interval=100):
        print(f'Using device: {self.device}')
        self.d_losses_real, self.d_losses_fake, self.g_losses = [], [], []
        self.d_accuracies_real, self.d_accuracies_fake = [], []
        current_datetime = datetime.datetime.now().strftime('%d-%m_%H-%M')

        save_path = f'./dc_wgan_gp_generated_samples_{current_datetime}/'
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(epochs):
            for i, (imgs, features) in enumerate(data_loader):
                imgs = imgs.view(-1, 3, 1000).to(self.device)
                features = features.float().to(self.device)
                real_imgs = imgs.float()

                # Train Discriminator
                self.optimiser_D.zero_grad()
                noise = torch.randn(imgs.size(0), self.noise_dim, device=self.device)
                fake_imgs = self.generator(noise, features).detach()
                real_loss = self.discriminator(real_imgs, features).mean()
                fake_loss = self.discriminator(fake_imgs, features).mean()
                gradient_penalty = self.compute_gradient_penalty(real_imgs.data, fake_imgs.data, features)
                d_loss = fake_loss - real_loss + gradient_penalty
                d_loss.backward()
                self.optimiser_D.step()

                # Train Generator every nth step
                if i % 5 == 0:

                    self.optimiser_G.zero_grad()
                    gen_imgs = self.generator(noise, features)
                    g_loss = -self.discriminator(gen_imgs, features).mean()
                    g_loss.backward()
                    self.optimiser_G.step()

                print(f"Epoch: {epoch}, Batch: {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

                if epoch % save_interval == 0:
                    with torch.no_grad():
                        sample_noise = torch.randn(self.batchsize, self.noise_dim, device=self.device)
                        sample_features = torch.randn(self.batchsize, self.feature_dim, device=self.device)
                        generated_samples = self.generator(sample_noise, sample_features)
                        np.save(os.path.join(save_path, f'generated_samples_epoch_{epoch}.npy'), generated_samples.cpu().numpy())
                
                self.d_losses_real.append(real_loss.item())
                self.d_losses_fake.append(fake_loss.item())
                self.g_losses.append(g_loss.item())

                d_accuracy_real = ((self.discriminator(real_imgs, features) > 0.5).float().mean()).item()
                d_accuracy_fake = ((self.discriminator(fake_imgs, features) < 0.5).float().mean()).item()
                self.d_accuracies_real.append(d_accuracy_real)
                self.d_accuracies_fake.append(d_accuracy_fake)

        torch.save(self.generator.state_dict(), f'./model_states/dc_wgan_gp_generator_{current_datetime}.pth')
        torch.save(self.discriminator.state_dict(), f'./model_states/dc_wgan_gp_discriminator_{current_datetime}.pth')

        plt.figure(figsize=(15, 5))

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

        os.makedirs('./training_plots', exist_ok=True)
        plt.savefig(f'./training_plots/training_metrics_epochs_{epochs}.png')
        plt.close()
        #return fake_imgs.detach()


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import mne
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import datetime

class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_dim=3, channels=3, time_steps=1000):
        super(Generator, self).__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.model = nn.Sequential(
            nn.Linear(noise_dim + feature_dim, 64),
            nn.LeakyReLU(0.02, True), 
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.02, True), 
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.02, True), 
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.02, True), 
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.02, True), 
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.02, True), 
            nn.BatchNorm1d(1024),
            nn.Linear(1024, channels * time_steps),
            nn.Tanh()
        )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noise, features):
        x = torch.cat((noise, features), dim=1)
        output = self.model(x)
        return output.view(-1, self.channels, self.time_steps)


class Discriminator(nn.Module):
    def __init__(self, feature_dim, channels, time_steps=1000):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.time_steps = time_steps

        self.model = nn.Sequential(
            nn.Linear(channels*time_steps+feature_dim, 1024),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.02, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, True)
        )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img, features):
        img_flat = img.view(img.size(0), -1)
        x = torch.cat((img_flat, features), dim=1)
        return self.model(x)

class WGAN:
    def __init__(self, channels, batchsize, noise_dim, feature_dim):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.channels = channels
        self.batchsize = batchsize
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim

        self.generator = Generator(noise_dim=noise_dim, feature_dim=feature_dim, channels=channels).to(self.device)
        self.discriminator = Discriminator(channels=channels, feature_dim=feature_dim).to(self.device)

        self.optimiser_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimiser_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.9, 0.999))
        

    def compute_gradient_penalty(self, real_images, fake_images, features):
        # Random weight term for interpolation between real and fake samples
        #print('real image shape: ', real_images.shape)
        alpha = torch.rand(real_images.size(0), 1, device=self.device)
        #print('alpha shape: ', alpha.shape)
        alpha = alpha.expand(real_images.size(0),real_images.size(1))

        # Get random interpolation between real and fake images
        #print('alpha shape after expand: ', alpha.shape)
        #print('Fake images shape ', fake_images.shape)
        fake_images_flat = fake_images.view(fake_images.size(0), -1)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images_flat).requires_grad_(True)
        decision = self.discriminator(interpolated, features)
        gradients = torch.autograd.grad(outputs=decision, inputs=interpolated,
                                        grad_outputs=torch.ones(decision.size(), device=self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, data_loader, epochs, lambda_gp=1, save_interval=100):
        d_losses, g_losses = [], []
        d_real_losses, d_fake_losses = [], []
        d_real_acc, d_fake_acc = [], []
        current_datetime = datetime.datetime.now().strftime('%d-%m_%H-%M')

        save_path = f'./Training_Samples/wgan_generated_samples_{current_datetime}'
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(epochs):
            d_loss_sum, g_loss_sum = 0.0, 0.0
            d_real_loss_sum, d_fake_loss_sum = 0.0, 0.0
            d_real_correct, d_fake_correct = 0, 0
            n_batches = 0
            g_updates = 0
            d_updates = 0

            for i, (imgs, features) in enumerate(data_loader):
                real_imgs = imgs.float().to(self.device)
                features = features.float().to(self.device)
                noise = torch.randn(imgs.size(0), self.noise_dim, device=self.device)
                gen_imgs = self.generator(noise, features)

                # Training Discriminator
                #if i % 100 == 0:
                self.optimiser_D.zero_grad()
                real_decision = self.discriminator(real_imgs, features).mean()
                fake_decision = self.discriminator(gen_imgs.detach(), features).mean()

                d_real_loss = -real_decision.mean()
                d_fake_loss = fake_decision.mean()

                d_real_correct += (real_decision < 0).sum().item()
                d_fake_correct += (fake_decision > 0).sum().item()

                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(real_imgs, gen_imgs, features) * lambda_gp
                d_loss = fake_decision - real_decision + gradient_penalty

                d_loss.backward(retain_graph=True)
                self.optimiser_D.step()

                d_loss_sum += d_loss.item()
                d_real_loss_sum += d_real_loss.item()
                d_fake_loss_sum += d_fake_loss.item()
                n_batches += 1

                # Training Generator
                if i % 10 == 0:
                    self.optimiser_G.zero_grad()
                    gen_loss = -self.discriminator(gen_imgs, features).mean()
                    gen_loss.backward()
                    self.optimiser_G.step()
                    g_loss_sum += gen_loss.item()
                    g_updates += 1

            avg_d_loss = d_loss_sum / n_batches
            avg_g_loss = g_loss_sum / n_batches
            avg_d_real_loss = d_real_loss_sum / n_batches
            avg_d_fake_loss = d_fake_loss_sum / n_batches
            avg_d_real_acc = d_real_correct / (real_imgs.size(0) * n_batches)
            avg_d_fake_acc = d_fake_correct / (real_imgs.size(0) * n_batches)

            d_losses.append(avg_d_loss)
            g_losses.append(avg_g_loss)
            d_real_losses.append(avg_d_real_loss)
            d_fake_losses.append(avg_d_fake_loss)
            d_real_acc.append(avg_d_real_acc)
            d_fake_acc.append(avg_d_fake_acc)

            print(f"Epoch: {epoch}, D Loss: {avg_d_loss}, G Loss: {avg_g_loss}")

            if epoch % save_interval == 0:
                with torch.no_grad():
                    sample_noise = torch.randn(self.batchsize, self.noise_dim, device=self.device)
                    sample_features = torch.randn(self.batchsize, self.feature_dim, device=self.device)
                    generated_samples = self.generator(sample_noise, sample_features)
                    np.save(os.path.join(save_path, f'generated_samples_epoch_{epoch}.npy'), generated_samples.cpu().numpy())

        torch.save(self.generator.state_dict(),f'./Model_States/WGAN_GP/wgan_generator_{current_datetime}.pth')
        #torch.save(self.discriminator.state_dict(),f'./Model_States/WGAN_GP/wgan_discriminator_{current_datetime}.pth')
        os.makedirs(save_path, exist_ok=True)
        # Plotting losses and accuracies
        plt.figure(figsize=(15, 5))

        plt.subplot(2, 2, 1)
        plt.plot(d_losses, label='D Loss')
        plt.plot(g_losses, label='G Loss')
        plt.title('Total Discriminator and Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(d_real_losses, label='D Real Loss')
        plt.plot(d_fake_losses, label='D Fake Loss')
        plt.title('Discriminator Real and Fake Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()

        os.makedirs('./Training_Plots/WGAN_GP_training_plots', exist_ok=True)
        plt.savefig(f'./Training_Plots/WGAN_GP_training_plots/training_losses_epochs_{epochs}_{current_datetime}.png')
        plt.show()
        plt.close()
      


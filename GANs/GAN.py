import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')
import mne
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

class Generator(nn.Module):
    def __init__(self, noise_dim=100, feature_dim=3, channels=3, time_steps=1000):
        super(Generator, self).__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.model = nn.Sequential(
            nn.Linear(noise_dim + feature_dim, 128),
            nn.LeakyReLU(0.02, inplace=True), #changed from 0.2
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.02, inplace=True), #changed from 0.2
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.02, inplace=True), #changed from 0.2
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.02, inplace=True), #changed from 0.2
            nn.BatchNorm1d(1024),
            nn.Linear(1024, channels * time_steps),#25 * 342),
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
        #image_size_flat = channels * 1000 #channels * 342
        #input_dim = image_size_flat + feature_dim

        self.model = nn.Sequential(
            #nn.Linear(input_dim, 512),
            nn.Linear(channels*time_steps+feature_dim, 512),
            nn.LeakyReLU(0.02, inplace=True), #changed from 0.2
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, inplace=True), #changed from 0.2
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img, features):
        img_flat = img.view(img.size(0), -1)
        print('img_flat shape: ', img_flat.shape)
        print('features shape: ', features.shape)
        x = torch.cat((img_flat, features), dim=1)
        print('x shape: ', x.shape,)
        return self.model(x)


class GAN:
    def __init__(self, channels=3, batchsize=50,  noise_dim=100, feature_dim=3):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.channels = channels
        self.batchsize = batchsize
        self.noise_dim = noise_dim  
        self.feature_dim = feature_dim

        self.generator = Generator(noise_dim=noise_dim, feature_dim=feature_dim, channels=channels).to(self.device)
        self.discriminator = Discriminator(channels=channels, feature_dim=feature_dim).to(self.device) #removed channels, may need to add back

        print(f"Number of parameters in discriminator: {len(list(self.discriminator.parameters()))}")

        self.optimiser_G = optim.Adam(self.generator.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.optimiser_D = optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

        self.loss = nn.BCELoss()

    def train(self, data_loader, epochs):
        #d_losses, g_losses = [], []
        d_losses, d_real_losses, d_fake_losses, g_losses = [], [], [], []
        d_real_accuracy, d_fake_accuracy = [], []  # Track accuracies
        lr_G = self.optimiser_G.param_groups[0]['lr']  
        lr_D = self.optimiser_D.param_groups[0]['lr']

        for epoch in range(epochs):
            d_loss_sum, g_loss_sum = 0.0, 0.0
            n_batches = 0

            for i, (imgs, features) in enumerate(data_loader):
                valid = torch.ones((imgs.size(0), 1), device=self.device)
                fake = torch.zeros((imgs.size(0), 1), device=self.device)

                real_imgs = imgs.float().to(self.device)
                features = features.float().to(self.device)

                # Generate fakes
                noise = torch.randn(imgs.size(0), self.noise_dim, device=self.device)
                gen_imgs = self.generator(noise, features)
                print(f'generated image shape: {gen_imgs.shape}')

                # Generator
                if i % 5 == 0:
                    self.optimiser_G.zero_grad()
                    #noise = torch.randn(imgs.size(0), self.noise_dim, device=self.device)
                    #gen_imgs = self.generator(noise, features)
                    g_loss = self.loss(self.discriminator(gen_imgs, features), valid)
                    g_loss.backward()
                    self.optimiser_G.step()
                    g_loss_sum += g_loss.item()
                    g_losses.append(g_loss.item())

                # Discriminator
                #if i % 100 == 0:
                self.optimiser_D.zero_grad()
                real_decision = self.discriminator(real_imgs, features)
                real_loss = self.loss(self.discriminator(real_imgs, features), valid)
                fake_decision = self.discriminator(gen_imgs.detach(), features)
                fake_loss = self.loss(self.discriminator(gen_imgs.detach(), features), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward(retain_graph=True)
                real_loss.backward(retain_graph=True)  # Retain computation graph for next backward pass
                fake_loss.backward()
                self.optimiser_D.step()

                # Calculate accuracies
                real_correct = (real_decision >= 0.5).float()
                fake_correct = (fake_decision < 0.5).float()
                d_real_acc = real_correct.mean().item()
                d_fake_acc = fake_correct.mean().item()
                d_real_accuracy.append(d_real_acc)
                d_fake_accuracy.append(d_fake_acc)

                d_loss_sum += d_loss.item()
                #g_loss_sum += g_loss.item()
                n_batches += 1

                #if i % sample_interval == 0:
            print(f"Epoch: {epoch}, Batch: {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, Real acc: {d_real_acc}, Fake acc: {d_fake_acc}")

            d_losses.append(d_loss_sum / n_batches)
            d_real_losses.append(real_loss.item())
            d_fake_losses.append(fake_loss.item())
            #g_losses.append(g_loss_sum / n_batches)


            #if epoch % sample_interval == 0:
            self.save_samples(epoch, gen_imgs)

            torch.save(self.generator.state_dict(), './gan_generator_model_final.pth')
            torch.save(self.discriminator.state_dict(), './gan_discriminator_model_final.pth')

        plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        #plt.plot(d_real_losses, label='Discriminator Real Loss')
        #plt.plot(d_fake_losses, label='Discriminator Fake Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Subplot for the accuracies
        plt.subplot(1, 2, 2)
        plt.plot(d_real_accuracy, label='Discriminator Real Accuracy')
        plt.plot(d_fake_accuracy, label='Discriminator Fake Accuracy')
        plt.title('Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
        os.makedirs('training_plots', exist_ok=True)
        plt.savefig(f'training_plots/training_losses_epochs_{epochs}_LRG_{lr_G}_LRD_{lr_D}.png')
        plt.close()


    '''def train(self, data_loader, epochs):

        d_losses = []
        g_losses = []

        for epoch in range(epochs):
            for i, (imgs, features) in enumerate(data_loader):
                valid = torch.ones((imgs.size(0), 1), device=self.device)
                fake = torch.zeros((imgs.size(0), 1), device=self.device)

                real_imgs = imgs.float().to(self.device)
                features = features.float().to(self.device)

                # Generator
                self.optimiser_G.zero_grad()
                noise = torch.randn(imgs.size(0), self.noise_dim, device=self.device)
                gen_imgs = self.generator(noise, features)
                g_loss = self.loss(self.discriminator(gen_imgs, features), valid)
                g_loss.backward()
                self.optimiser_G.step()

                # Discriminator
                self.optimiser_D.zero_grad()
                real_loss = self.loss(self.discriminator(real_imgs, features), valid)
                fake_loss = self.loss(self.discriminator(gen_imgs.detach(), features), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimiser_D.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

                #if i % sample_interval == 0:
                print(f"Epoch: {epoch}, Batch: {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

            #if epoch % sample_interval == 0:
                self.save_samples(epoch, gen_imgs)
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.title('Training Losses')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()'''

    def save_samples(self, epoch, gen_imgs, sfreq=250):
        # Ensure the directories exist
        os.makedirs('generated_data_new', exist_ok=True)
        os.makedirs("gen_eeg", exist_ok=True)

        # Save the generated EEG data as a numpy array
        data = gen_imgs.data.cpu().numpy()  # Shape: (batch_size, channels, time_steps)
        file_path = f'generated_data_new/epoch_{epoch}.npy'
        try:
            np.save(file_path, data)
            print(f"Saved EEG data to {file_path}")
        except Exception as e:
            print(f"Error saving EEG data to {file_path}: {e}")

        # For visualization, use the first sample in the batch
        sample_data = data[0]  # Shape: (channels, time_steps)

        # Create MNE info structure, assuming the channel type is EEG
        info = mne.create_info(
            ch_names=[f'EEG {i+1}' for i in range(sample_data.shape[0])],
            sfreq=sfreq,
            ch_types=['eeg'] * sample_data.shape[0]
        )

        # Create the RawArray object
        raw = mne.io.RawArray(sample_data, info)

        # Plotting the EEG data using MNE
        try:
            fig = raw.plot(duration=5, n_channels=sample_data.shape[0], show=False, title=f'Generated EEG Data at Epoch {epoch}')
            image_path = f"images/epoch_{epoch}.png"
            fig.savefig(image_path)
            print(f"Saved plot to {image_path}")
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot to {image_path}: {e}")

    '''def save_samples(self, epoch, gen_imgs):
        os.makedirs('generated_data', exist_ok=True)
        os.makedirs("images", exist_ok=True)
        data = gen_imgs.data.cpu().numpy()
        np.save(f'generated_data/epoch_{epoch}.npy', data)
        img_grid = make_grid(gen_imgs.data[:25], nrow=5, normalize=True)
        plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy(), interpolation='nearest')
        plt.title(f"Epoch {epoch}")
        plt.savefig(f"images/epoch_{epoch}.png")
        plt.close()'''


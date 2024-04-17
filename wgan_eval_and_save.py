import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from GANs.WGAN import Generator
from data_loader import EEGFeatureDataset

# Set the directory paths
data_dir = '/Users/umairarshad/SHU/or_data/mat'
feature_dir = '/Users/umairarshad/SHU/or_data/csp_features'
save_base_dir = '/Users/umairarshad/Projects/CSP_DC_CGAN/Generated_Samples_Eval'  # Base directory to save the generated data

# Subjects and sessions
subjects = range(21, 26)  # Subjects 21 to 25
sessions = range(1, 6)   # Sessions 1 to 5

# Initialize the Generator
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
generator = Generator(noise_dim=1000, feature_dim=3, channels=3, time_steps=1000).to(device)
generator.load_state_dict(torch.load('./Model_States/WGAN_GP/wgan_generator_17-04_13-38.pth', map_location=device))
generator.eval()

# Loop through each subject and session
for subject in subjects:
    for session in sessions:
        # Update the save_dir for each subject and session
        save_dir = f"{save_base_dir}/WGAN_Eval_17-04_13-38"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Load the dataset for the current subject and session
        test_dataset = EEGFeatureDataset(data_dir, feature_dir, [subject], [session])
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

        # Generate and save samples
        with torch.no_grad():
            for i, (real_data, features) in enumerate(test_loader):
                noise = torch.randn(features.size(0), 1000, device=device)
                generated_samples = generator(noise, features.to(device))

                # Save the generated samples to a file
                #np.save(f"{save_dir}/GEN_sub-{subject}ses-{session}.npy", generated_samples.cpu().numpy()) #sub-013_ses-03_task_motorimagery_events.tsv
                np.save(f"{save_dir}/GEN_sub-{subject:03d}_ses-{session:02d}_specific.npy", generated_samples.cpu().numpy())
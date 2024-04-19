import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import datetime

from GANs.WGAN import Generator
from data_loader import EEGFeatureDataset

#directory paths
data_dir = '/Users/umairarshad/SHU/or_data/mat'
feature_dir = '/Users/umairarshad/SHU/or_data/csp_features'
save_base_dir = '/Users/umairarshad/Projects/CSP_WGAN/Generated_Samples_Eval'  #base directory to save the generated data
current_datetime = datetime.datetime.now().strftime('%d-%m_%H-%M')

#choose ubjects and sessions
subjects = range(1, 26)
sessions = range(1, 6)

#initialize Generator
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
generator = Generator(noise_dim=1000, feature_dim=3, channels=3, time_steps=1000).to(device)
generator.load_state_dict(torch.load('./Model_States/WGAN_GP/wgan_generator_17-04_13-38.pth', map_location=device))
generator.eval()

#loop through
for subject in subjects:
    for session in sessions:
        #update the save_dir
        save_dir = f"{save_base_dir}/WGAN_Eval_17-04_13-38_{current_datetime}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #load the dataset for current subject and session
        test_dataset = EEGFeatureDataset(data_dir, feature_dir, [subject], [session])
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

        #generate and save samples
        with torch.no_grad():
            for i, (real_data, features) in enumerate(test_loader):
                noise = torch.randn(features.size(0), 1000, device=device)
                generated_samples = generator(noise, features.to(device))

                
                np.save(f"{save_dir}/GEN_sub-{subject:03d}_ses-{session:02d}.npy", generated_samples.cpu().numpy())
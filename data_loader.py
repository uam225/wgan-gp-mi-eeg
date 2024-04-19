import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torch

class EEGFeatureDataset(Dataset):
    def __init__(self, data_dir, feature_dir, subjects, sessions):
        self.data_samples = []
        self.feature_samples = []

        for subj in subjects:
            for sess in sessions:
                data_path = os.path.join(data_dir, f'sub-{subj:03d}_ses-{sess:02d}_task_motorimagery_eeg.mat')
                feature_path = os.path.join(feature_dir, f'csp_features_sub{subj}_sess{sess}.mat')

                #load EEG data
                mat_data = sio.loadmat(data_path)
                all_data = mat_data['data']
                cz_index, c3_index, c4_index = 11, 12, 13
                data = all_data[:, [cz_index, c3_index, c4_index], :]

                #load features
                mat_features = sio.loadmat(feature_path)
                features = mat_features['csp_features']

                min_trials = min(data.shape[0], features.shape[0])
                eeg_shape = data.shape[1] * data.shape[2]
                features_no = features.shape[1]


                for trial in range(min_trials):
                    #normalise each EEG data trial
                    trial_data = data[trial]
                    trial_data_min = np.min(trial_data)
                    trial_data_max = np.max(trial_data)
                    normalized_trial_data = 2 * (trial_data - trial_data_min) / (trial_data_max - trial_data_min) - 1
                    flattened_data = normalized_trial_data.flatten()
                    self.data_samples.append(flattened_data)

                    #normalise each feature vector
                    trial_features = features[trial]
                    trial_features = (trial_features - np.mean(trial_features)) / np.std(trial_features)
                    self.feature_samples.append(trial_features)

        
        self.data_samples = np.array(self.data_samples)
        self.feature_samples = np.array(self.feature_samples)

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        data_sample = torch.tensor(self.data_samples[idx], dtype=torch.float32)
        feature_sample = torch.tensor(self.feature_samples[idx], dtype=torch.float32)
        return data_sample, feature_sample
#global normalisation 
'''import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torch

class EEGFeatureDataset(Dataset):
    def __init__(self, data_dir, feature_dir, subjects, sessions, feature_fold=10):
        self.data_samples = []
        self.feature_samples = []

        for subj in subjects:
            for sess in sessions:
                data_path = os.path.join(data_dir, f'sub-{subj:03d}_ses-{sess:02d}_task_motorimagery_eeg.mat')
                feature_path = os.path.join(feature_dir, f'csp_features_sub{subj}_sess{sess}.mat')

                # Load EEG data
                mat_data = sio.loadmat(data_path)
                all_data = mat_data['data'] 
                cz_index, c3_index, c4_index = 11, 12, 13
                data = all_data[:,[cz_index, c3_index, c4_index], :] 
                data_min = np.min(data)
                data_max = np.max(data)
                data = 2 * (data - data_min) / (data_max - data_min) - 1
                
                # Load features
                mat_features = sio.loadmat(feature_path)
                features = mat_features['csp_features']
                features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

                min_trials = min(data.shape[0], features.shape[0])
                eeg_shape = data.shape[1] * data.shape[2]
                features_no = features.shape[1]
                self.data_samples = np.ones((min_trials,eeg_shape))
                self.feature_samples = np.ones((min_trials,features_no))

                for trial in range(min_trials):
                    flattened_data = data[trial].flatten()
                    self.data_samples[trial,:] = flattened_data
                    self.feature_samples[trial,:] = features[trial,:]

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        data_sample = torch.tensor(self.data_samples[idx], dtype=torch.float32)  # Convert to float32
        feature_sample = torch.tensor(self.feature_samples[idx], dtype=torch.float32)  # Convert to float32
        return data_sample, feature_sample'''

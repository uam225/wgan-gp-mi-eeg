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

                # Load EEG data
                mat_data = sio.loadmat(data_path)
                all_data = mat_data['data']
                cz_index, c3_index, c4_index = 11, 12, 13
                data = all_data[:, [cz_index, c3_index, c4_index], :]

                # Load features
                mat_features = sio.loadmat(feature_path)
                features = mat_features['csp_features']

                min_trials = min(data.shape[0], features.shape[0])
                eeg_shape = data.shape[1] * data.shape[2]
                features_no = features.shape[1]

                '''for trial in range(min_trials):
                    #normalise each EEG data trial using z-score normalization
                    trial_data = data[trial]
                    trial_mean = np.mean(trial_data)
                    trial_std = np.std(trial_data)
                    normalized_trial_data = (trial_data - trial_mean) / trial_std
                    flattened_data = normalized_trial_data.flatten()
                    self.data_samples.append(flattened_data)

                    #normalise each feature vector (already using z-score normalization)
                    trial_features = features[trial]
                    trial_features = (trial_features - np.mean(trial_features)) / np.std(trial_features)
                    self.feature_samples.append(trial_features)'''

                for trial in range(min_trials):
                    #normalize each EEG data trial
                    trial_data = data[trial]
                    trial_data_min = np.min(trial_data)
                    trial_data_max = np.max(trial_data)
                    normalized_trial_data = 2 * (trial_data - trial_data_min) / (trial_data_max - trial_data_min) - 1
                    flattened_data = normalized_trial_data.flatten()
                    self.data_samples.append(flattened_data)

                    #normalize each feature vector
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



#unused code
'''
data_dir = '/Users/umairarshad/SHU/or_data/mat'
feature_dir = '/Users/umairarshad/SHU/or_data/features'  
subjects = range(6, 11)  
sessions = range(1, 6)  

dataset = EEGFeatureDataset(data_dir, feature_dir, subjects, sessions)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

type(dataset)

mat_data = sio.loadmat('/Users/umairarshad/SHU/or_data/mat/sub-001_ses-01_task_motorimagery_eeg.mat')
data = mat_data['data']
print(data.shape)

features = np.load(feature_dir + '/sub_4_ses_1_fold_1_train_features.npy')
features.shape
min_trials = min(data.shape[0], features.shape[0]) 
data_samples_check = np.zeros((min_trials,32000))#[]
features_samples_check = np.zeros((min_trials,10))#[]

for trial in range(min_trials):
    flattened_data = data[trial].flatten()
    data_samples_check[trial,:] = flattened_data
    features_samples_check[trial, :] = features[trial,:]
    #data_samples_check.append(flattened_data)
    #features_samples_check.append(features[trial])
print(flattened_data.shape)
print(data_samples_check.shape)
print(features_samples_check.shape)'''
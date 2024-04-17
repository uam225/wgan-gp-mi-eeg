import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def normalize_to_same_scale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def load_and_normalize_data(subject_id, session_id, epoch):
    # File paths based on provided naming conventions
    original_data_path = f'/Users/umairarshad/SHU/or_data/mat/sub-{subject_id}_ses-{session_id}_task_motorimagery_eeg.mat'
    generated_data_path = f'./wgan_generated_samples_10-04_10-54/generated_samples_epoch_{epoch}.npy'

    # Load the original EEG data
    eeg_data = scipy.io.loadmat(original_data_path)['data']

    # Load the generated data
    generated_data = np.load(generated_data_path)

    # Normalize the data
    # Extracting only the channels Cz, C3 and C4
    eeg_data_normalized = np.array([normalize_to_same_scale(eeg_data[:, i, :]) for i in [10, 11, 12]])
    
    # Assuming generated_data has shape [batches, channels, data points], average over batches
    generated_data_avg = np.mean(generated_data, axis=0)
    # Normalizing the first three channels of the generated data as they correspond to Cz, C3, C4
    generated_data_normalized = np.array([normalize_to_same_scale(generated_data_avg[i, :]) for i in range(3)])

    return eeg_data_normalized, generated_data_normalized

def plot_data(eeg_data, generated_data, channels, sample_index=0):
    channel_names = ['Cz', 'C3', 'C4']  # Names of the channels in the original data
    plt.figure(figsize=(15, 10))
    
    for i, channel_index in enumerate(channels):
        plt.subplot(len(channels), 1, i+1)
        original_label = f'Original Data ({channel_names[i]})'
        generated_label = f'Generated Data (Ch {i+1})'
        plt.plot(eeg_data[i, sample_index, :], label=original_label, color='navy')
        plt.plot(generated_data[i, :], label=generated_label, color='orange')
        plt.title(f'{channel_names[i]} Original vs. Generated')
        plt.xlabel('Time points')
        plt.ylabel('Amplitude')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
subject_id = '002'  # Subject identifier
session_id = '01'  # Session identifier
epoch = '500'  # Epoch number for generated data
channels = [11, 12, 13]  # Indices for the channels in the original data (Cz, C3, C4)

eeg_data, generated_data = load_and_normalize_data(subject_id, session_id, epoch)
plot_data(eeg_data, generated_data, channels)



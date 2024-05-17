import numpy as np
import librosa
import os
from torch.utils.data import Dataset

import os
import random


def get_dataset_split(root_dir, categories, split_ratio=0.8):
    data = []
    labels = []
    for label, category in enumerate(categories):
        category_path = os.path.join(root_dir, category)
        for subject_id in os.listdir(category_path):
            subject_path = os.path.join(category_path, subject_id)
            if os.path.isdir(subject_path):
                audio_files = [os.path.join(subject_path, f) for f in os.listdir(subject_path) if f.endswith('.wav')]
                data.extend(audio_files)
                labels.extend([label] * len(audio_files))
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    split_point = int(len(data) * split_ratio)
    return (data[:split_point], labels[:split_point]), (data[split_point:], labels[split_point:])


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs


def pad_or_trim(features, max_len=500):
    if features.shape[1] > max_len:
        return features[:, :max_len]
    elif features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        return np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        return features


class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, max_len=500):
        self.file_paths = file_paths
        self.labels = labels
        self.max_len = max_len
        self.features = [pad_or_trim(extract_features(file_path), max_len) for file_path in file_paths]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        features = self.features[idx].T
        label = self.labels[idx]
        return features, label

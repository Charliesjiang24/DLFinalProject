import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
import random


class AudioClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        

        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        x = x.transpose(1, 2) 
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :]) 
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
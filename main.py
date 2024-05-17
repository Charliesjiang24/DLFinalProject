from train import train_model
from data_loader import get_dataset_split, AudioDataset
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    root_dir = '.'
    categories = ['HC', 'MDD']
    (train_files, train_labels), (val_files, val_labels) = get_dataset_split(root_dir, categories)
    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(train_loader, val_loader, device)

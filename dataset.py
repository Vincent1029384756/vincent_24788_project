"""
Data loader
"""
import torch
from torch.utils.data import Dataset, DataLoader

class ShapeNetDataset(Dataset):
    def __init__(self, split='train'):
        # ensure the passeed split names are valid
        assert split in ['train', 'val', 'test', 'trainval']
        path = rf'data\ShapeNet\processed\air_{split}.pt'
        data = torch.load(path, weights_only=False)

        self.points = data[0]['pos'] # (5042175, 3) position of all points
        self.labels = data[0]['y'] # (5042175, ) all part labels (wing, body, tail, engine)
        self.slices = data[1]['pos'] # indices where each individual shape starts and ends

        # filter out empty shapes
        self.valid_indices = [
            i for i in range(len(self.slices) - 1)
            if self.slices[i+1] - self.slices[i] > 0
        ]

    def __len__(self):
        # return number of shapes
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        start = self.slices[real_idx]
        end = self.slices[real_idx + 1]
        points = self.points[start:end]
        labels = self.labels[start:end]

        num_points = 2048
        n = points.shape[0]
        choice = torch.randint(0, n, (num_points,))
        points = points[choice]
        labels = labels[choice]

        return points, labels
            
def get_dataloader(split='train', batch_size=32, shuffle=True):
    dataset = ShapeNetDataset(split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == 'train' else False # only shuffle the training data
    )
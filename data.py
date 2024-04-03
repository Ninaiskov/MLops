import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class CorruptMNIST(Dataset):
    def __init__(self, dataset, trainset_idx, split='trainval', transform=None, target_transform=None):
        self.main_path = '/Users/Nina/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU_MLOPS'
        self.data_path = os.path.join(self.main_path, 'data', dataset)
        if split == 'trainval':
            self.X = torch.load(os.path.join(self.data_path, 'train_images_'+str(trainset_idx)+'.pt'))
            self.y = torch.load(os.path.join(self.data_path, 'train_target_'+str(trainset_idx)+'.pt'))
        elif split == 'test':
            self.X = torch.load(os.path.join(self.data_path, 'test_images.pt'))
            self.y = torch.load(os.path.join(self.data_path, 'test_target.pt'))
        else:
            raise ValueError('split must be one of "trainval" or "test"')    
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx].unsqueeze(0)
        label = self.y[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def mnist(dataset, trainset_idx, batch_size, shuffle, split):
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784)
    transform = transforms.Normalize((0.5,), (0.5,))
    target_transform = None
    if split == 'trainval':
        trainval_dataset = CorruptMNIST(dataset=dataset, trainset_idx=trainset_idx, split=split, 
                                    transform=transform, target_transform=target_transform)
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [0.8, 0.2])
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        
        return trainloader, valloader
    
    elif split == 'test':
        test_dataset = CorruptMNIST(dataset=dataset, trainset_idx=None, split=split, 
                                    transform=transform, target_transform=target_transform)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return testloader
    else:
        raise ValueError('split must be one of "trainval" or "test"')


import os

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader


DATA_DIR = "../input/foodcvfetisov/Food/"


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.45),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(p=0.45),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'test']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in ['train', 'test']
}

class_names = image_datasets['train'].classes
from torch.autograd import Variable
from torch.utils.data import random_split
from torchvision import datasets, transforms
import os
# You can download the data from this link : https://www.kaggle.com/datasets/greatgamedota/ffhq-face-data-set
def prepare_data(path, split_size=0.8, batch_size=16):
    transform = transforms.Compose([transforms.Resize(255),
                                transforms.ToTensor()
                               ])
    dataset = datasets.ImageFolder(path, transform=transform)
    lengths = [int(len(dataset)*split_size), int(len(dataset)*(1-split_size)]
    train_dataset, val_dataset = random_split(dataset, lengths)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader

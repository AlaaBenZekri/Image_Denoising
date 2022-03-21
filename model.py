import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Conv2d, ConvTranspose2d, MaxPool2d, AdaptiveAvgPool2d, Module, BatchNorm2d, Sigmoid, Dropout
import torch.optim as optim

class AutoEncoders(nn.Module):
    def __init__(self,
                 in_channels=3,
                 kernel_size=(3,3),
                 ):
        super(autoencoders, self).__init__()
        self.encoder = Sequential(
            Conv2d(in_channels, 32, kernel_size = kernel_size, padding = "same"),
            ReLU(),
            MaxPool2d((2,2), padding = 0),
            Conv2d(32, 64, kernel_size = kernel_size, padding = "same"),
            ReLU(),
            MaxPool2d((2,2), padding = 0),
            Conv2d(64, 128, kernel_size = kernel_size, padding = "same"),
            ReLU(),
            MaxPool2d((2,2), padding = 0)
        )
        self.decoder = Sequential(
            ConvTranspose2d(128, 128, kernel_size = kernel_size, stride = 2, padding = 0),
            ReLU(),
            ConvTranspose2d(128, 64, kernel_size = kernel_size, stride = 2, padding = 0),
            ReLU(),
            ConvTranspose2d(64, 32, kernel_size = kernel_size, stride = 2, padding = 0),
            ReLU(),
            ConvTranspose2d(32, in_channels, kernel_size = kernel_size, stride = 1, padding = 1),
            Sigmoid()
        )
        
    def forward(self, images):
        x = self.encoder(images)
        x = self.decoder(x)
        return x

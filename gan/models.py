import torch
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #conv1
            nn.Conv2d(input_channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            #conv2
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            #conv3
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            #conv4
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            #conv5
            nn.Conv2d(1024, 1, 4, 1, 0)
        )
    
    def forward(self, x):
        return self.main(x)


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        self.main = nn.Sequential(
            #Transosed conv1
            nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            #Transosed conv2
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            #Transosed conv3
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            #Transosed conv4
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            #Transosed conv4
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)
    


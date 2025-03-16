# small generator model for GAN

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_channels=3, num_features=32):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )

        self.down1 = nn.Conv2d(num_features, num_features * 2, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.final = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.up1(x)
        x = self.up2(x)
        return torch.tanh(self.final(x))



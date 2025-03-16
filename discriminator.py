# small discriminator model for GAN, predicting a grid of real/fake values (patch-wise classification)
# this way it's not classifying the entire image as real/fake, but individual patches
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256]):
        super().__init__()

        # Initial layer (no InstanceNorm)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Downsampling
        layers = []
        in_channels = features[0]
        for feature in features:
            layers.append(ConvBlock(in_channels, feature, stride=2))
            in_channels = feature
        
        # Final conv layer (PatchGAN output)
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(self.initial(x)))  # Patch-based output
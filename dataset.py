# defines a class for a pixel and scenery image dataset

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

class PixelSceneryDataset(Dataset):
    def __init__(self, root_pixel, root_scenery, transform=None):
        self.root_pixel = root_pixel
        self.root_scenery = root_scenery
        self.transform = transform

        self.pixel_images = [p for p in glob.glob(os.path.join(root_pixel, "*")) if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        self.scenery_images = [s for s in glob.glob(os.path.join(root_scenery, "*")) if s.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    def __len__(self):
        return min(len(self.pixel_images), len(self.scenery_images))

    def __getitem__(self, index):

        index = index % len(self)

        pixel_img = Image.open(self.pixel_images[index]).convert("RGB")
        scenery_img = Image.open(self.scenery_images[index]).convert("RGB")

        if self.transform:
            pixel_img = self.transform(pixel_img)
            scenery_img = self.transform(scenery_img)

        return scenery_img, pixel_img
# train a cycleGAN model to translate between scenery images and pixel art images

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

from discriminator import Discriminator
from generator import Generator


def train_one_epoch(discriminator_scenery, discriminator_pixel, 
                    generator_scenery, generator_pixel, dataloader, 
                    optimizer_discriminator, optimizer_generator,
                    lambda_cycle):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for index, (scenery, pixel) in tqdm(dataloader):
        pixel = pixel.to(device)
        scenery = scenery.to(device)

        # train the discriminators
        with torch.cuda.amp.autocast():
            fake_scenery = generator_scenery(pixel)
            discriminator_scenery_real = discriminator_scenery(scenery)
            discriminator_scenery_fake = discriminator_scenery(fake_scenery.detach())

            discriminator_scenery_loss_real = nn.MSELoss(discriminator_scenery_real, torch.ones_like(discriminator_scenery_real))
            discriminator_scenery_loss_fake = nn.MSELoss(discriminator_scenery_fake, torch.zeros_like(discriminator_scenery_fake))
            discriminator_scenery_loss = (discriminator_scenery_loss_real + discriminator_scenery_loss_fake) / 2
            
            fake_pixel = generator_pixel(scenery)
            discriminator_pixel_real = discriminator_pixel(pixel)
            discriminator_pixel_fake = discriminator_pixel(fake_pixel.detach())

            discriminator_pixel_loss_real = nn.MSELoss(discriminator_pixel_real, torch.ones_like(discriminator_pixel_real))
            discriminator_pixel_loss_fake = nn.MSELoss(discriminator_pixel_fake, torch.zeros_like(discriminator_pixel_fake))
            discriminator_pixel_loss = (discriminator_pixel_loss_real + discriminator_pixel_loss_fake) / 2

            total_discriminator_loss = discriminator_scenery_loss + discriminator_pixel_loss

        optimizer_discriminator.zero_grad()
        torch.cuda.amp.GradScaler().scale(total_discriminator_loss).backward()
        torch.cuda.amp.GradScaler().step(optimizer_discriminator)
        torch.cuda.amp.GradScaler().update()

        # train the generators
        with torch.cuda.amp.autocast():
            discriminator_scenery_fake = discriminator_scenery(fake_scenery)
            discriminator_pixel_fake = discriminator_pixel(fake_pixel)

            generator_loss_scenery = nn.MSELoss(discriminator_scenery_fake, torch.ones_like(discriminator_scenery_fake))
            generator_loss_pixel = nn.MSELoss(discriminator_pixel_fake, torch.ones_like(discriminator_pixel_fake))

            total_generator_loss = generator_loss_scenery + generator_loss_pixel

            # cycle consistency loss
            cycle_scenery = generator_scenery(fake_pixel)
            cycle_pixel = generator_pixel(fake_scenery)

            cycle_loss_scenery = nn.L1Loss(cycle_scenery, scenery)
            cycle_loss_pixel = nn.L1Loss(cycle_pixel, pixel)

            total_cycle_loss = cycle_loss_scenery + cycle_loss_pixel

            total_generator_loss = total_generator_loss + lambda_cycle * total_cycle_loss

        optimizer_generator.zero_grad()
        torch.cuda.amp.GradScaler().scale(total_generator_loss).backward()
        torch.cuda.amp.GradScaler().step(optimizer_generator)
        torch.cuda.amp.GradScaler().update()

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyperparameters
    num_epochs = 50
    lambda_cycle = 10
    learning_rate = 2e-4
    batch_size = 1

    discriminator_scenery = Discriminator().to(device)
    discriminator_pixel = Discriminator().to(device)
    generator_scenery = Generator().to(device)
    generator_pixel = Generator().to(device)

    optimizer_discriminator = optim.Adam(list(discriminator_scenery.parameters()) + list(discriminator_pixel.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_generator = optim.Adam(list(generator_scenery.parameters()) + list(generator_pixel.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    scenery_dataset = torchvision.datasets.ImageFolder(root="data/scenery", transform=transform)
    pixel_dataset = torchvision.datasets.ImageFolder(root="data/pixel", transform=transform)

    scenery_dataloader = DataLoader(scenery_dataset, batch_size=batch_size, shuffle=True)
    pixel_dataloader = DataLoader(pixel_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        train_one_epoch(discriminator_scenery, discriminator_pixel, generator_scenery, generator_pixel, scenery_dataloader, pixel_dataloader, optimizer_discriminator, optimizer_generator, lambda_cycle)

if __name__ == "__main__":
    train()
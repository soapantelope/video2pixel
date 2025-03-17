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
from dataset import PixelSceneryDataset

def train_one_epoch(discriminator_scenery, discriminator_pixel, 
                    generator_scenery, generator_pixel, dataloader, 
                    optimizer_discriminator, optimizer_generator,
                    lambda_cycle, mse_loss, l1_loss, scaler):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    average_discriminator_loss = 0
    average_generator_loss = 0

    for (scenery, pixel) in tqdm(dataloader):
        pixel = pixel.to(device)
        scenery = scenery.to(device)

        # train the discriminators
        with torch.amp.autocast('cuda'):
            fake_scenery = generator_scenery(pixel)
            discriminator_scenery_real = discriminator_scenery(scenery)
            discriminator_scenery_fake = discriminator_scenery(fake_scenery.detach())

            discriminator_scenery_loss_real = mse_loss(discriminator_scenery_real, torch.ones_like(discriminator_scenery_real))
            discriminator_scenery_loss_fake = mse_loss(discriminator_scenery_fake, torch.zeros_like(discriminator_scenery_fake))
            discriminator_scenery_loss = (discriminator_scenery_loss_real + discriminator_scenery_loss_fake) / 2
            
            fake_pixel = generator_pixel(scenery)
            discriminator_pixel_real = discriminator_pixel(pixel)
            discriminator_pixel_fake = discriminator_pixel(fake_pixel.detach())

            discriminator_pixel_loss_real = mse_loss(discriminator_pixel_real, torch.ones_like(discriminator_pixel_real))
            discriminator_pixel_loss_fake = mse_loss(discriminator_pixel_fake, torch.zeros_like(discriminator_pixel_fake))
            discriminator_pixel_loss = (discriminator_pixel_loss_real + discriminator_pixel_loss_fake) / 2

            total_discriminator_loss = discriminator_scenery_loss + discriminator_pixel_loss

            average_discriminator_loss += total_discriminator_loss.item() / len(dataloader)

        optimizer_discriminator.zero_grad()
        scaler.scale(total_discriminator_loss).backward()
        scaler.step(optimizer_discriminator)
        scaler.update()

        # train the generators
        with torch.amp.autocast('cuda'):
            discriminator_scenery_fake = discriminator_scenery(fake_scenery)
            discriminator_pixel_fake = discriminator_pixel(fake_pixel)

            generator_loss_scenery = mse_loss(discriminator_scenery_fake, torch.ones_like(discriminator_scenery_fake))
            generator_loss_pixel =  mse_loss(discriminator_pixel_fake, torch.ones_like(discriminator_pixel_fake))

            total_generator_loss = generator_loss_scenery + generator_loss_pixel

            # cycle consistency loss
            cycle_scenery = generator_scenery(fake_pixel)
            cycle_pixel = generator_pixel(fake_scenery)

            cycle_loss_scenery = l1_loss(cycle_scenery, scenery)
            cycle_loss_pixel = l1_loss(cycle_pixel, pixel)

            total_cycle_loss = cycle_loss_scenery + cycle_loss_pixel

            total_generator_loss = total_generator_loss + lambda_cycle * total_cycle_loss

            average_generator_loss += total_generator_loss.item() / len(dataloader)

        optimizer_generator.zero_grad()
        scaler.scale(total_generator_loss).backward()
        scaler.step(optimizer_generator)
        scaler.update()

    print(f"Discriminator loss: {average_discriminator_loss}, Generator loss: {average_generator_loss}")
    return average_discriminator_loss, average_generator_loss

def save_model(discriminator_scenery, discriminator_pixel, generator_scenery, generator_pixel, optimizer_discriminator, optimizer_generator):
    torch.save(discriminator_scenery.state_dict(), "discriminator_scenery.pth")
    torch.save(discriminator_pixel.state_dict(), "discriminator_pixel.pth")
    torch.save(generator_scenery.state_dict(), "generator_scenery.pth")
    torch.save(generator_pixel.state_dict(), "generator_pixel.pth")

    # save optimizers
    torch.save(optimizer_discriminator.state_dict(), "optimizer_discriminator.pth")
    torch.save(optimizer_generator.state_dict(), "optimizer_generator.pth")

def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor image.

    Args:
        tensor (torch.Tensor): Normalized tensor of shape (C, H, W).
        mean (list): List of mean values for each channel.
        std (list): List of std values for each channel.

    Returns:
        torch.Tensor: Denormalized tensor of shape (C, H, W).
    """
    mean = torch.tensor(mean).to(tensor.device).view(3, 1, 1)
    std = torch.tensor(std).to(tensor.device).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor

def validate(generator_scenery, generator_pixel, transform, epoch, num_images=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_dataset = PixelSceneryDataset("data/validation/scenery", "data/validation/pixel", transform=transform)

    # pick 3 random images from the validation set and save the generated images
    # make sure to de-normalize the images before saving
    indices = torch.randint(0, len(validation_dataset), (num_images,))
    for i in indices:
        scenery, pixel = validation_dataset[i]
        scenery = scenery.unsqueeze(0).to(device)
        pixel = pixel.unsqueeze(0).to(device)

        fake_scenery = generator_scenery(pixel)
        fake_pixel = generator_pixel(scenery)

        scenery = denormalize(scenery, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        pixel = denormalize(pixel, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        fake_scenery = denormalize(fake_scenery, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        fake_pixel = denormalize(fake_pixel, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        torchvision.utils.save_image(scenery, f"epoch_{epoch}_scenery_{i}.png")
        torchvision.utils.save_image(pixel, f"epoch_{epoch}_pixel_{i}.png")
        torchvision.utils.save_image(fake_scenery, f"epoch_{epoch}_fake_scenery_{i}.png")
        torchvision.utils.save_image(fake_pixel, f"epoch_{epoch}_fake_pixel_{i}.png")

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyperparameters
    num_epochs = 100
    lambda_cycle = 5
    learning_rate = 2e-4
    batch_size = 4

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    discriminator_scenery = Discriminator().to(device)
    discriminator_pixel = Discriminator().to(device)
    generator_scenery = Generator().to(device)
    generator_pixel = Generator().to(device)

    optimizer_discriminator = optim.Adam(list(discriminator_scenery.parameters()) + list(discriminator_pixel.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_generator = optim.Adam(list(generator_scenery.parameters()) + list(generator_pixel.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = PixelSceneryDataset("data/scenery", "data/pixel", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_discriminator_loss = float("inf")
    best_generator_loss = float("inf")
    not_improved_epochs = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        average_discriminator_loss, average_generator_loss = train_one_epoch(discriminator_scenery, discriminator_pixel, 
                        generator_scenery, generator_pixel, dataloader,
                        optimizer_discriminator, optimizer_generator, 
                        lambda_cycle, mse_loss, l1_loss, scaler)
        
        if average_discriminator_loss < best_discriminator_loss and average_generator_loss < best_generator_loss:
            best_discriminator_loss = average_discriminator_loss
            best_generator_loss = average_generator_loss
        elif average_discriminator_loss > best_discriminator_loss and average_generator_loss > best_generator_loss:
            not_improved_epochs += 1
        else:
            not_improved_epochs = 0

        if not_improved_epochs >= 3:
            print("Early stopping")
            break

        if epoch % 2 == 0:
            # save the model
            print("Saving the model")
            save_model(discriminator_scenery, discriminator_pixel, generator_scenery, generator_pixel, optimizer_discriminator, optimizer_generator)
            # save some generated images
            validate(generator_scenery, generator_pixel, transform, epoch, 3)
               
    # save the model
    save_model(discriminator_scenery, discriminator_pixel, generator_scenery, generator_pixel, optimizer_discriminator, optimizer_generator)

    # load the model
    
    # discriminator_scenery.load_state_dict(torch.load("discriminator_scenery.pth"))
    # discriminator_pixel.load_state_dict(torch.load("discriminator_pixel.pth"))
    # generator_scenery.load_state_dict(torch.load("generator_scenery.pth"))
    # generator_pixel.load_state_dict(torch.load("generator_pixel.pth"))

    validate(generator_scenery, generator_pixel, transform, num_epochs, 10)

if __name__ == "__main__":
    # train()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator_scenery = Generator().to(device)
    generator_pixel = Generator().to(device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # load mdoels
    generator_scenery.load_state_dict(torch.load("generator_scenery.pth"), strict=False)
    generator_pixel.load_state_dict(torch.load("generator_pixel.pth"), strict=False)

    validate(generator_scenery, generator_pixel, transform, 1, 20)

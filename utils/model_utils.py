import os
import torch

def save_model(generator, discriminator, epoch, save_dir='models/'):
    os.makedirs(save_dir, exist_ok=True)
    
    generator_path = os.path.join(save_dir, f'generator_epoch_{epoch}.pth')
    discriminator_path = os.path.join(save_dir, f'discriminator_epoch_{epoch}.pth')
    
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    
    print(f"Models saved at epoch {epoch}:")
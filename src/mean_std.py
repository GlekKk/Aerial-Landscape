from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(CURRENT_DIR, '..', 'data', 'processed', 'train')
IMG_SIZE = 224
BATCH_SIZE = 32

def calculate_mean_std():
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor() 
    ])

    dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    print(f"MEAN = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"STD = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

if __name__ == '__main__':
    calculate_mean_std()
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = 224
BATCH_SIZE = 64

def get_data_loaders(data_dir, model_name="simplecnn"):

    if model_name == 'resnet18':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
    else:
        MEAN = [0.3780, 0.3927, 0.3442]
        STD = [0.1300, 0.1173, 0.1118]

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),    
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)]),
        'test': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)])
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], 
                      batch_size = BATCH_SIZE, 
                      shuffle = (x == 'train'), 
                      num_workers = 0)
        for x in ['train', 'val', 'test']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

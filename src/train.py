import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import json

from dataset import get_data_loaders
from model import SimpleCNN 
from model import get_resnet

CURRENT_MODEL = 'resnet18'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data', 'processed')
NUM_CLASSES = 15
if CURRENT_MODEL == 'simplecnn':
    NUM_EPOCHS = 60
else:
    NUM_EPOCHS = 30
LEARNING_RATE = 0.001 # For SGD 0.01

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataloaders, dataset_sizes, class_names = get_data_loaders(DATA_DIR, CURRENT_MODEL)
    print(f"Found classes: {len(class_names)}")
    print(f"Size of train: {dataset_sizes['train']} images, val: {dataset_sizes['val']} images")

    if CURRENT_MODEL == 'resnet18':
        model = get_resnet(num_classes=NUM_CLASSES)
    else:
        model = SimpleCNN(num_classes=NUM_CLASSES)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Trying different optimizer

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                scheduler.step(epoch_loss)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'\nTrain completed: {best_acc:.4f}')
    
    MODELS_DIR = os.path.join(CURRENT_DIR, '..', 'models')
    HISTORY_DIR = os.path.join(CURRENT_DIR, '..', 'results', 'history_json')
    
    model_file_name = f'{CURRENT_MODEL}_model2.pth'
    history_file_name = f'{CURRENT_MODEL}_history2.json'
    
    model.load_state_dict(best_model_wts)
    MODEL_PATH = os.path.join(MODELS_DIR, model_file_name)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model was saved as '{model_file_name}'")

    HISTORY_PATH = os.path.join(HISTORY_DIR, '..', history_file_name)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    main()
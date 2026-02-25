import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

from dataset import get_data_loaders
from model import SimpleCNN
from model import get_resnet

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data', 'processed')

MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'simplecnn_model.pth')
# MODEL_PATH = os.path.join(CURRENT_DIR, '..', 'resnet_model.pth') - for ResNet18
NUM_CLASSES = 15

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We are using: {device}")
    
    dataloaders, _, class_names = get_data_loaders(DATA_DIR)
    test_loader = dataloaders['test']
    
    model = SimpleCNN(num_classes=NUM_CLASSES)
    # model = get_resnet(num_classes=NUM_CLASSES) - for ResNet18
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion matrix (SimpleCNN)', fontsize=16)
    # plt.title('Confusion matrix (ResNet)', fontsize=16) - for ResNet18
    plt.xlabel('Prediction', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(CURRENT_DIR, '..', 'simplecnn_confusion_matrix.png')
    # save_path = os.path.join(CURRENT_DIR, '..', 'resnet_confusion_matrix.png') - for ResNet18
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == '__main__':
    evaluate()
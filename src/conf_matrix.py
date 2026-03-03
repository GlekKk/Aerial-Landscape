import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

from dataset import get_data_loaders
from model import SimpleCNN
from model import get_resnet

CURRENT_MODEL = 'resnet18'

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data', 'processed')
NUM_CLASSES = 15

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We are using: {device}")
    
    dataloaders, _, class_names = get_data_loaders(DATA_DIR, CURRENT_MODEL)
    test_loader = dataloaders['test']

    MODELS_DIR = os.path.join(CURRENT_DIR, '..', 'models')
    CONF_DIR = os.path.join(CURRENT_DIR, '..', 'results', 'confusion_matrix')

    model_filename = f'{CURRENT_MODEL}_model1.pth'
    MODEL_PATH = os.path.join(MODELS_DIR, model_filename)

    if CURRENT_MODEL == 'resnet18':
        model = get_resnet(num_classes=NUM_CLASSES)
    else:
        model = SimpleCNN(num_classes=NUM_CLASSES)
        
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
            
    print("\n" + "="*60)
    print(f"   CLASSIFICATION REPORT FOR {CURRENT_MODEL.upper()} v. 1.0")
    print("="*60)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    print("="*60 + "\n")

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion matrix ({CURRENT_MODEL.upper()}) v. 1.0', fontsize=16)
    plt.xlabel('Prediction', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(CONF_DIR, f'{CURRENT_MODEL}_confusion_matrix1.png')
    
    plt.savefig(save_path, dpi=300)
    print(f"Matrix saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    evaluate()
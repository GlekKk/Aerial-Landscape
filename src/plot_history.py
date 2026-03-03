import json
import matplotlib.pyplot as plt
import os

CURRENT_MODEL = 'resnet18'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

HISTORY_DIR = os.path.join(CURRENT_DIR, '..', 'results', 'history_json')
PLOT_DIR = os.path.join(CURRENT_DIR, '..', 'results', 'plot_history')
HISTORY_PATH = os.path.join(HISTORY_DIR, f'{CURRENT_MODEL}_history1.json')

def plot_metrics():

    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o', color='red')
    plt.title(f'Loss ({CURRENT_MODEL.upper()}) v. 1.0', fontsize=14) 
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o', color='blue')
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o', color='red')
    plt.title(f'Accuracy ({CURRENT_MODEL.upper()}) v. 1.0', fontsize=14) 
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, f'{CURRENT_MODEL}_history_plot1.png')
    plt.savefig(save_path, dpi=300)
    
    plt.show()

if __name__ == '__main__':
    plot_metrics()
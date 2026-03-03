import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from model import SimpleCNN
from model import get_resnet

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data', 'processed', 'train')

MODELS_DIR = os.path.join(CURRENT_DIR, '..', 'models')

# MODEL_PATH = os.path.join(MODELS_DIR, 'simplecnn_model4.pth') - for SimpleCNN
MODEL_PATH = os.path.join(MODELS_DIR, 'resnet18_model3.pth')

IMAGE_PATH = os.path.join(CURRENT_DIR, '..', 'test_image.jpg') 

def predict():

    class_names = sorted([d.name for d in os.scandir(DATA_DIR) if d.is_dir()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = SimpleCNN(num_classes=len(class_names)) - for SimpleCNN
    model = get_resnet(num_classes=len(class_names))
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(IMAGE_PATH).convert('RGB')
    
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        
        probabilities = F.softmax(outputs, dim=1)
        
        top_probs, top_classes = torch.topk(probabilities, 3)
    
    for i in range(3):
        c_index = top_classes[0][i].item()
        c_prob = top_probs[0][i].item() * 100
        c_name = class_names[c_index]
        
        if i == 0:
            print(f"Object on the img is: {c_name.upper()} ({c_prob:.2f}%)")
            print("-" * 40)
            print("Other options:")
        else:
            print(f"   • {c_name}: {c_prob:.2f}%")
    print("="*40 + "\n")

if __name__ == '__main__':
    predict()
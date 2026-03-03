import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):

    def __init__(self, num_classes):

        super(SimpleCNN, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2) v 1.0
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # v. 2.0
        ) 

        self.features2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2) v 1.0
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # v. 2.0
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Added new layer

        self.features4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),            
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),             
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
def get_resnet(num_classes):

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

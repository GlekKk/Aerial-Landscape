# Aerial Landscape Image Classification

This project focuses on classifying aerial landscape images into 15 distinct categories using PyTorch. It documents the complete engineering process: from building and debugging a custom Convolutional Neural Network (SimpleCNN) to implementing transfer learning with a pre-trained ResNet18 model. 

The final architecture achieves a 97.0% validation accuracy, demonstrating high reliability even on visually similar and complex landscape classes.

## Project Structure

The repository is organized to separate source code, saved models, and evaluation results:

```text
project_root/
├── data/                  # Raw and processed datasets split into train, val, and test
├── models/                # Saved PyTorch model weights (.pth)
├── results/               # Training logs and visual evaluation metrics
│   ├── confusion_matrix/  # Generated confusion matrix plots
│   ├── history_json/      # Raw training history data
│   └── plot_history/      # Accuracy and loss graphs
├── src/                   # Main source code
│   ├── dataset.py         # Data loading, augmentation, and transformations
│   ├── model.py           # SimpleCNN and ResNet18 architectures
│   ├── train.py           # Main training loop with learning rate scheduling
│   ├── predict.py         # Inference script for classifying new images
│   ├── conf_matrix.py     # Script to generate classification reports and matrices
│   └── mean_std.py        # Utility to calculate dataset normalization statistics
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```

## Experiment Log

SimpleCNN v. 1.0:
We started with a custom 3-layer SimpleCNN. Initially trained for 50 epochs using the Adam optimizer with learning rate = 0.001, 
the model achieved around 82.0% accuracy. 

SimpleCNN v. 2.0:
To force the network to learn robust textures, we added a Dropout(0.5) layer in the classifier 
and implemented overlapping pooling (MaxPool2d(kernel_size=3, stride=2, padding=1)). 
We also added a ReduceLROnPlateau scheduler and extended the training to 60 epochs. 
This architectural update completely eliminated the overfitting, 
stabilizing our validation metrics and bringing the accuracy to a reliable 83.4%.
During this architectural upgrade, the increased computational overhead exposed a critical memory leak on Windows 
caused by multiprocess data loading. We debugged and resolved this by setting num_workers = 0 in the DataLoader


SimpleCNN v. 3.0:
Realizing the 3-layer architecture had hit its capacity limit, we added a 4th convolutional block, 
increasing the feature channels from 64 to 128. This allowed the model to extract deeper spatial context. 
The accuracy surged by 6%, reaching 89.0%. 
The confusion matrix cleared up significantly for difficult, visually similar landscapes.

SimpleCNN v. 4.0
We temporarily switched our optimizer from Adam to SGD with Momentum (0.9) and a higher starting learning rate (0.01). 
While SGD converged much faster—hitting a plateau around epoch 38—it began to overfit slightly during the final epochs.
This experiment validated our hyperparameter tuning strategy but confirmed that Adam provided better late-stage stability for this specific custom architecture.

ResNet18 v. 1.0:

Trained for 15 epochs using Adam (learning rate = 0.001) but without a dynamic learning rate scheduler, 
the model briefly hit a peak accuracy of ~95%. 
However, the validation loss was highly unstable and chaotic, 
proving that a static learning rate was too aggressive for the fine-tuning phase

ResNet18 v. 2.0:
Trained for 15 epochs with Adam and our dynamic learning rate scheduler, the model achieved 93.4% accuracy. 
The training curve became stable, and the network successfully separated complex mixed landscapes

ResNet18 v. 3.0:
We extended the training duration to 30 epochs. The result was 97.0% validation accuracy, 
with the model making absolutely zero classification errors on highly complex classes like Ports, Parking, and Residential areas.
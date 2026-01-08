# Train the MobileNetV2 model on PlantVillage dataset
import sys
import xml.dom.minidom

# Fix for Python 3.12 compatibility
if not hasattr(xml.dom.minidom, 'NodeFilter'):
    class NodeFilter:
        FILTER_ACCEPT = 1
        FILTER_REJECT = 2
        FILTER_SKIP = 3
        SHOW_ALL = 0xFFFFFFFF
    xml.dom.minidom.NodeFilter = NodeFilter
import os
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ========================
# CONFIGURATION
# ========================
class Config:
    # Dataset paths
    DATASET_PATH = r"C:\Users\ayush\Desktop\Machine Learning\PlantVillage"
    
    # Model parameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    
    # Split ratios (Handled by folder structure or splitting logic)
    VAL_SPLIT = 0.15
    
    # Output paths
    MODEL_SAVE_PATH = "models/crop_disease_model.pth"
    LABELS_SAVE_PATH = "models/class_labels.json"
    HISTORY_SAVE_PATH = "models/training_history.json"
    
    # Device configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = Config()

# ========================
# DATA PREPARATION
# ========================

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def create_dataloaders():
    """
    Create train, validation, and test dataloaders with augmentation
    """
    print("🔄 Creating data loaders...")
    
    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load full dataset
    full_dataset = datasets.ImageFolder(config.DATASET_PATH)
    
    # Filter out "PlantVillage" class if it exists (Recursive folder issue)
    if "PlantVillage" in full_dataset.classes:
        print("⚠️  Detected recursive 'PlantVillage' folder. Excluding it from training.")
        target_idx = full_dataset.class_to_idx["PlantVillage"]
        # Filter samples
        full_dataset.samples = [s for s in full_dataset.samples if s[1] != target_idx]
        # Update classes
        full_dataset.classes.remove("PlantVillage")
        del full_dataset.class_to_idx["PlantVillage"]
        # Remap indices
        old_to_new = {v: i for i, (k, v) in enumerate(full_dataset.class_to_idx.items())}
        full_dataset.samples = [(s[0], old_to_new[s[1]]) for s in full_dataset.samples]
        
    class_names = full_dataset.classes
    
    # Split into train and validation
    train_size = int((1 - config.VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms (Hack: Since random_split wraps dataset, we need to handle transforms differently or subclass)
    # For simplicity, we'll apply transforms at batch time or use a wrapper class.
    # Here we will use a custom wrapper for simplicity.
    
    
    train_data = TransformedDataset(train_dataset, transform=data_transforms['train'])
    val_data = TransformedDataset(val_dataset, transform=data_transforms['val'])

    # Create dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }
    
    dataset_sizes = {'train': len(train_data), 'val': len(val_data)}
    
    # Save class labels
    class_labels = {i: name for i, name in enumerate(class_names)}
    os.makedirs(os.path.dirname(config.LABELS_SAVE_PATH), exist_ok=True)
    with open(config.LABELS_SAVE_PATH, 'w') as f:
        json.dump(class_labels, f, indent=4)
    
    print(f"✅ Found {dataset_sizes['train']} training images")
    print(f"✅ Found {dataset_sizes['val']} validation images")
    print(f"✅ Number of classes: {len(class_names)}")
    print(f"✅ Class labels saved to {config.LABELS_SAVE_PATH}")
    
    return dataloaders, dataset_sizes, class_names

# ========================
# MODEL ARCHITECTURE
# ========================
def build_model(num_classes):
    """
    Build MobileNetV2 model
    """
    print("🏗️  Downloading MobileNetV2 pretrained model...")
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier
    # MobileNetV2 classifier is usually:
    # (classifier): Sequential(
    #   (0): Dropout(p=0.2, inplace=False)
    #   (1): Linear(in_features=1280, out_features=1000, bias=True)
    # )
    
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model.to(config.DEVICE)

# ========================
# TRAINING
# ========================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # Using simple loop to avoid tqdm dependency issues for now
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                print("✅ Model checkpoint saved.")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save history
    with open(config.HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=4)
        
    return model, history

# ========================
# EVALUATION & VISUALIZATION
# ========================
def visualize_history(history):
    print("� Plotting training history...")
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("✅ Training plots saved.")

# ========================
# MAIN
# ========================
def main():
    print("="*80)
    print("🌿 CROP DISEASE CLASSIFICATION - MODEL TRAINING (PYTORCH)")
    print("="*80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"�️  Device: {config.DEVICE}")
    print("="*80)
    
    # Create dataloaders
    dataloaders, dataset_sizes, class_names = create_dataloaders()
    
    # Build model
    model = build_model(len(class_names))
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train
    model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=config.EPOCHS)
    
    # Visualize
    visualize_history(history)
    
    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"💾 Model saved to: {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()

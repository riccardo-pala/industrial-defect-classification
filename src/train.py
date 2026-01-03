import os
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Paths and Hyperparameters
# ----------------------------
DATA_DIR = Path("data/processed")  # train/val/test inside
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 6  # NEU dataset
MODEL_SAVE_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("=" * 60)
logger.info("TRAINING CONFIGURATION")
logger.info("=" * 60)
logger.info(f"Device: {DEVICE}")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Number of epochs: {NUM_EPOCHS}")
logger.info(f"Learning rate: {LEARNING_RATE}")
logger.info(f"Number of classes: {NUM_CLASSES}")
logger.info(f"Model save path: {MODEL_SAVE_PATH}")

# ----------------------------
# Data Transforms & Augmentation
# ----------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ----------------------------
# Datasets & DataLoaders
# ----------------------------
train_dataset = datasets.ImageFolder(DATA_DIR/"train", transform=train_transforms)
val_dataset = datasets.ImageFolder(DATA_DIR/"val", transform=val_test_transforms)
test_dataset = datasets.ImageFolder(DATA_DIR/"test", transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

logger.info("=" * 60)
logger.info("DATASET LOADED")
logger.info("=" * 60)
logger.info(f"Training samples: {len(train_dataset)} (batches: {len(train_loader)})")
logger.info(f"Validation samples: {len(val_dataset)} (batches: {len(val_loader)})")
logger.info(f"Test samples: {len(test_dataset)} (batches: {len(test_loader)})")

# ----------------------------
# Model (Transfer Learning)
# ----------------------------
model = models.efficientnet_b0(pretrained=True)

# Replace classifier head
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

logger.info("=" * 60)
logger.info("MODEL INITIALIZED")
logger.info("=" * 60)
logger.info(f"Model: EfficientNet-B0 (pretrained)")
logger.info(f"Output classes: {NUM_CLASSES}")
logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ----------------------------
# Loss & Optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

logger.info(f"Loss function: CrossEntropyLoss")
logger.info(f"Optimizer: Adam (lr={LEARNING_RATE})")
logger.info("=" * 60)
logger.info("STARTING TRAINING")
logger.info("=" * 60)

# ----------------------------
# Training Loop
# ----------------------------
best_val_f1 = 0.0

# ----------------------------
# Early Stopping Parameters
# ----------------------------
patience = 5
counter = 0

for epoch in range(NUM_EPOCHS):
    logger.info(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}] Starting...")
    model.train()
    train_preds, train_labels = [], []
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_f1 = f1_score(train_labels, train_preds, average='macro')
    train_acc = accuracy_score(train_labels, train_preds)
    train_loss /= len(train_loader.dataset)
    
    logger.info(f"  [Train] Loss: {train_loss:.4f} | F1-Score: {train_f1:.4f} | Accuracy: {train_acc:.4f}")

    # ----------------------------
    # Validation
    # ----------------------------
    logger.info(f"  [Validation] Evaluating on validation set...")
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_acc = accuracy_score(val_labels, val_preds)

    logger.info(f"  [Validation] Loss: {val_loss:.4f} | F1-Score: {val_f1:.4f} | Accuracy: {val_acc:.4f}")

    # Save best model and handle early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("=> Best model saved")
        counter = 0  # reset counter
    else:
        counter += 1
        print(f"No improvement for {counter}/{patience} epochs")
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break  # exit training loop

# ----------------------------
# Test Evaluation
# ----------------------------
logger.info("=" * 60)
logger.info("LOADING BEST MODEL FOR TEST EVALUATION")
logger.info("=" * 60)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
logger.info(f"Model loaded from: {MODEL_SAVE_PATH}")

logger.info("\nEvaluating on test set...")
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_f1 = f1_score(test_labels, test_preds, average='macro')
test_acc = accuracy_score(test_labels, test_preds)

logger.info("\n" + "=" * 60)
logger.info("TEST RESULTS")
logger.info("=" * 60)
logger.info(f"Test F1-Score: {test_f1:.4f}")
logger.info(f"Test Accuracy: {test_acc:.4f}")
logger.info("=" * 60)
logger.info("TRAINING COMPLETED SUCCESSFULLY")
logger.info("=" * 60)

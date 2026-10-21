import os
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report, 
    ConfusionMatrixDisplay
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Get model
def get_model(model_name):
    return ResNet18_Weights.DEFAULT if model_name == 'ResNet18' else ResNet50_Weights.DEFAULT

# Transforms
def transform(model_name, IMAGE_SIZE):
    weights = get_model(model_name)

    # Use ImageNet normalization because we're fine-tuning pretrained ResNet
    imagenet_mean = weights.transforms().mean
    imagenet_std = weights.transforms().std

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, eval_transform


# Load datasets
def load_data(DATA_DIR, BATCH_SIZE, NUM_WORKERS, train_transform, eval_transform):
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "val"),
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "test"),
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("Class names:", train_dataset.classes)
    print("Class to index:", train_dataset.class_to_idx)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    return train_loader, val_loader, test_loader, test_dataset

# Build model
def build_model(model_name, NUM_CLASSES, LEARNING_RATE):
    weights = get_model(model_name)
    model = models.resnet18(weights = weights) if model_name == 'ResNet18' else models.resnet50(weights = weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    return model, criterion, optimizer


# Train / eval helpers
def run_one_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            if training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "precision": epoch_precision,
        "recall": epoch_recall,
        "f1": epoch_f1,
        "labels": all_labels,
        "preds": all_preds,
    }


# Training loop
def training(model, train_loader, criterion, optimizer, val_loader, MODEL_SAVE_PATH, NUM_EPOCHS):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}  # ADD


    for epoch in range(NUM_EPOCHS):
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_one_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print(
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['acc']:.4f} | "
            f"Precision: {train_metrics['precision']:.4f} | "
            f"Recall: {train_metrics['recall']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )
        print(
            f"Val   | Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['acc']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model to {MODEL_SAVE_PATH}")
    
    return best_model_wts, history


# Final test evaluation
def test_eval(model, save_dir, best_model_wts, test_loader, criterion, test_dataset):
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(best_model_wts)

    test_metrics = run_one_epoch(model, test_loader, criterion)

    print("\n=== Test Results ===")
    print(f"Loss      : {test_metrics['loss']:.4f}")
    print(f"Accuracy  : {test_metrics['acc']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    print(f"F1-score  : {test_metrics['f1']:.4f}")

    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    print("\nConfusion Matrix:")
    print(cm)

    # Save the confusion matrix as heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
    disp.plot()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()

    print("\nClassification Report:")
    print(
        classification_report(
            test_metrics["labels"],
            test_metrics["preds"],
            target_names=test_dataset.classes,
            zero_division=0
        )
    )
    return cm

# Plot learning curve
def plot_learning_curves(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.legend()

    ax2.plot(epochs, history["train_f1"], label="Train")
    ax2.plot(epochs, history["val_f1"], label="Val")
    ax2.set_title("F1 Score"); ax2.legend()

    plt.savefig(os.path.join(save_dir, "learning_curves.png"), bbox_inches="tight")
    plt.close()

# Complete resNet model pipeline
def resNet_helper(model_name, data_dir, save_dir, model_save_path, num_classes, batch_size, num_epochs, num_workers, image_size):
    DATA_DIR = data_dir   
    SAVE_DIR = save_dir
    MODEL_SAVE_PATH = model_save_path

    # Config
    NUM_CLASSES = num_classes
    BATCH_SIZE = batch_size
    NUM_EPOCHS = num_epochs
    LEARNING_RATE = 1e-4
    NUM_WORKERS = num_workers
    IMAGE_SIZE = image_size

    # Transform model
    train_transform, eval_transform = transform(model_name, IMAGE_SIZE)

    # Load data
    train_loader, val_loader, test_loader, test_dataset = load_data(DATA_DIR, BATCH_SIZE, NUM_WORKERS, train_transform, eval_transform)

    # Build model
    model, criterion, optimizer = build_model(model_name, NUM_CLASSES, LEARNING_RATE)

    # Train the model
    best_model_wts, history = training(model, train_loader, criterion, optimizer, val_loader, MODEL_SAVE_PATH, NUM_EPOCHS)

    # Test Eval
    cm = test_eval(model, SAVE_DIR, best_model_wts, test_loader, criterion, test_dataset)

    plot_learning_curves(history, SAVE_DIR)
import torch
import torch.nn as nn
from sklearn.metrics import precision_score
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5):
    model.to(device)
    best_precision = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate
        precision = evaluate_model(model, val_loader, device)
        if precision > best_precision:
            best_precision = precision
            best_model_state = model.state_dict()

    return best_model_state, best_precision

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy())

    return precision_score(all_labels, all_preds)

import os
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from scripts.data_loader import TemporalLobeDataset
from scripts.model import SimpleCNN
from scripts.train_eval import train_model
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load the full subject-label CSV
csv_df = pd.read_csv('train.csv')
all_subjects = csv_df['Subject_ID'].values
all_labels = csv_df['Label'].values

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter tuning space
learning_rates = [0.001, 0.0005]
batch_sizes = [16, 32]

# Track best model
best_overall_precision = 0.0
best_model_state = None
best_params = {}

# Start hyperparameter tuning
for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"\n🔍 Testing: lr={lr}, batch_size={batch_size}")
        fold_precisions = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(all_subjects, all_labels)):
            print(f"\n📂 Fold {fold+1}")

            # Filter subjects for current fold
            train_subjects = set(all_subjects[train_idx])
            val_subjects = set(all_subjects[val_idx])

            train_df = csv_df[csv_df['Subject_ID'].isin(train_subjects)]
            val_df = csv_df[csv_df['Subject_ID'].isin(val_subjects)]

            # Save temporary fold CSVs
            train_df.to_csv("fold_train.csv", index=False)
            val_df.to_csv("fold_val.csv", index=False)

            # Create datasets/loaders
            train_dataset = TemporalLobeDataset("fold_train.csv", "data", transform)
            val_dataset = TemporalLobeDataset("fold_val.csv", "data", transform)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Initialize model
            model = SimpleCNN()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()

            model_state, precision = train_model(model, train_loader, val_loader, optimizer, criterion, device)
            fold_precisions.append(precision)
            print(f"✔️ Fold {fold+1} precision: {precision:.4f}")

        avg_precision = sum(fold_precisions) / len(fold_precisions)
        print(f"\n📊 Average Precision for lr={lr}, batch_size={batch_size}: {avg_precision:.4f}")

        if avg_precision > best_overall_precision:
            best_overall_precision = avg_precision
            best_model_state = model_state
            best_params = {'lr': lr, 'batch_size': batch_size}

# Save best model
os.makedirs("models", exist_ok=True)
torch.save(best_model_state, "models/best_model.pth")
print(f"\n✅ Best Hyperparameters: lr={best_params['lr']}, batch_size={best_params['batch_size']}")
print(f"🎯 Best Cross-Validated Precision: {best_overall_precision:.4f}")

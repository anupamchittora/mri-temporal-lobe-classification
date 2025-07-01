import torch
from torch.utils.data import DataLoader
from scripts.data_loader import TemporalLobeDataset
from scripts.model import SimpleCNN
from torchvision import transforms
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load test data
test_dataset = TemporalLobeDataset("test.csv", "data", transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load best model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

# Inference
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Convert to numpy
y_true = np.array(all_labels)
y_pred = np.array(all_preds)

# Metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Unhealthy"])

# Print Scores
print(f"✅ Test Accuracy  : {acc:.4f}")
print(f"🎯 Test Precision : {precision:.4f}")
print(f"🔁 Test Recall    : {recall:.4f}")
print(f"📐 Test F1-Score  : {f1:.4f}")

# Plot


# Plot confusion matrix
plt.figure(figsize=(6, 6), num="Confusion Matrix - Test Set")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Unhealthy"])
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("Confusion Matrix - Test Set")
plt.grid(False)
plt.tight_layout()
plt.show()

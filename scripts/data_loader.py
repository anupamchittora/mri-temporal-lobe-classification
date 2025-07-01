import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class TemporalLobeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        for _, row in self.labels_df.iterrows():
            folder = os.path.join(root_dir, f"{row['Subject_ID']}_temporal_lobe")
            label = int(row['Label'])
            if not os.path.exists(folder):
                print(f"⚠️ Missing folder: {folder}")
                continue
            image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
            if not image_files:
                print(f"⚠️ No image files in: {folder}")
            for file in image_files:
                self.samples.append((os.path.join(folder, file), label))

        print(f"✅ Loaded {len(self.samples)} image slices from {csv_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)  # Needed for BCEWithLogitsLoss

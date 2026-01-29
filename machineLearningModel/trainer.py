import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import multiprocessing

# =========================
# USER CONFIG
# =========================

# Root directory - the D:\ drive where all your data is located
DATA_ROOT = r"D:\\"

# CSV files are on the D:\ drive directly
TRAIN_CSV = os.path.join(DATA_ROOT, "train_visualCheXbert.csv")
VALID_CSV = os.path.join(DATA_ROOT, "CheXpert-v1.0 batch 1 (validate & csv)", "valid.csv")

# Image root is D:\ because paths in CSV are relative to this
IMAGE_ROOT = DATA_ROOT

BATCH_SIZE = 32  # Adjust based on your RAM
EPOCHS = 5
LR = 1e-4

# OPTIMIZED FOR USB DISK BOTTLENECK
# Fewer workers to reduce disk thrashing on slow USB drive
NUM_WORKERS = 2  # Lower because USB is slow

print(f"⚠️  WARNING: Data is on USB drive (slow)")
print(f"   For 5-10x speed improvement, move data to C:\\ drive (NVMe SSD)")
print(f"   Using {NUM_WORKERS} workers (optimized for slow disk)")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Try to use Intel GPU if available
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"✓ Using Intel GPU acceleration")
except ImportError:
    print(f"   Using CPU (install torch-directml for Intel GPU support)")

LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

# Patient ranges for each batch (based on actual data)
BATCH_RANGES = [
    (1, 21513, "CheXpert-v1.0 batch 2 (train 1)"),
    (21514, 43017, "CheXpert-v1.0 batch 3 (train 2)"),
    (43018, 64540, "CheXpert-v1.0 batch 4 (train 3)")
]

# =========================
# DATASET
# =========================

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.is_train = is_train

        # Keep only the labels we care about
        self.df = self.df[["Path"] + LABELS]

        # Replace uncertain labels (-1) with 0
        self.df[LABELS] = self.df[LABELS].replace(-1, 0)

        # Replace NaN with 0
        self.df[LABELS] = self.df[LABELS].fillna(0)

    def __len__(self):
        return len(self.df)

    def _get_batch_folder(self, patient_num):
        """Determine which batch folder a patient belongs to"""
        for min_patient, max_patient, batch_folder in BATCH_RANGES:
            if min_patient <= patient_num <= max_patient:
                return batch_folder
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        csv_path = row["Path"]
        csv_path = csv_path.replace("/", "\\")
        
        if self.is_train:
            parts = csv_path.split("\\")
            if len(parts) >= 4:
                patient_id = parts[2]
                study_and_file = "\\".join(parts[3:])
                
                try:
                    patient_num = int(patient_id.replace("patient", ""))
                except ValueError:
                    raise ValueError(f"Could not parse patient number from: {patient_id}")
                
                batch_folder = self._get_batch_folder(patient_num)
                
                if batch_folder is None:
                    raise ValueError(f"Patient {patient_num} is outside all known batch ranges")
                
                img_path = os.path.join(
                    self.image_root, 
                    batch_folder, 
                    batch_folder,
                    patient_id,
                    study_and_file
                )
            else:
                img_path = os.path.join(self.image_root, csv_path)
        else:
            parts = csv_path.split("\\")
            if len(parts) >= 4:
                patient_and_rest = "\\".join(parts[2:])
                img_path = os.path.join(
                    self.image_root,
                    "CheXpert-v1.0 batch 1 (validate & csv)",
                    "valid",
                    patient_and_rest
                )
            else:
                img_path = os.path.join(self.image_root, csv_path)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"\nERROR: Could not find image at {img_path}")
            raise

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[LABELS].values.astype("float32"))

        return image, labels

# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# LOADERS
# =========================

train_dataset = CheXpertDataset(TRAIN_CSV, IMAGE_ROOT, train_transform, is_train=True)
valid_dataset = CheXpertDataset(VALID_CSV, IMAGE_ROOT, valid_transform, is_train=False)

# Optimized for slow USB disk
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,  # Low for USB
    pin_memory=True,
    persistent_workers=True if NUM_WORKERS > 0 else False,
    prefetch_factor=8 if NUM_WORKERS > 0 else None  # Higher prefetch to compensate
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True if NUM_WORKERS > 0 else False,
    prefetch_factor=8 if NUM_WORKERS > 0 else None
)


# =========================
# MODEL
# =========================

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, len(LABELS))
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =========================
# TRAINING LOOP
# =========================

def train_epoch(loader):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def validate_epoch(loader):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(loader)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print(f"\nUsing device: {DEVICE}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    print("\n" + "="*60)
    print("💡 PERFORMANCE TIP:")
    print("="*60)
    print("Your data is on a USB drive (very slow).")
    print("To speed up training 5-10x:")
    print("1. Copy data to C:\\ drive (your fast NVMe SSD)")
    print("2. Install Intel GPU support: pip install torch-directml")
    print("="*60)
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        train_loss = train_epoch(train_loader)
        val_loss = validate_epoch(valid_loader)

        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")

        checkpoint_path = f"chexpert_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"   ✓ Saved: {checkpoint_path}")

    print("\n" + "="*60)
    print("🎉 Training complete!")
    print("="*60)
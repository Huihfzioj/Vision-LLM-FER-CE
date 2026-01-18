import os
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd

# --- Normalization Statistics ---
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# --- Augmentation Strategies ---
def get_transforms():
    """Returns a dictionary of transforms for different rarity levels + validation."""
    
    ultra_rare = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)  
    ])

    rare = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)  
    ])

    frequent = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return {
        "ultra_rare": ultra_rare,
        "rare": rare,
        "frequent": frequent,
        "val": val
    }

# --- Training Dataset (with Rarity Augmentation) ---
class RAFCETrainDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.class_counts = df["compound_emotion"].value_counts()
        self.transforms = get_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        
        # Ensure extension logic matches your preprocessing
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name += ".jpg"
            
        img_path = os.path.join(self.img_dir, img_name)
        
        # Robust loading
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fallback: create black image or raise useful error
            # For now, let's assume valid paths or crash to warn user
            raise FileNotFoundError(f"Image not found at {img_path}")

        label = int(row["compound_emotion"])
        count = self.class_counts.get(label, 0)

        # Choose augmentation based on rarity
        if count < 30:
            transform = self.transforms['ultra_rare']
        elif count < 200:
            transform = self.transforms['rare']
        else:
            transform = self.transforms['frequent']

        image = transform(image)
        return image, label

# --- Validation Dataset (Deterministic) ---
class RAFCEValDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = get_transforms()['val']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name += ".jpg"

        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
             raise FileNotFoundError(f"Image not found at {img_path}")
             
        label = int(row["compound_emotion"])
        image = self.transform(image)
        
        return image, label

# --- Weighted Sampler ---
def get_weighted_sampler(df):
    class_counts = df["compound_emotion"].value_counts().sort_index()
    # Avoid division by zero if a class is missing
    class_weights = 1.0 / (class_counts + 1e-6) 
    sample_weights = df["compound_emotion"].map(class_weights)
    
    # Handle NaN weights if map failed
    sample_weights = sample_weights.fillna(0)

    sampler = WeightedRandomSampler(
        weights=sample_weights.values,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

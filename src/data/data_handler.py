import os
import pandas as pd
from torchvision import transforms
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch

class DataHandler:
    def __init__(self, repo_id="mikkoim/aquamonitor-jyu", dataset_root="./dataset"):
        self.repo_id = repo_id
        self.dataset_root = dataset_root
        self.class_map = None
        self.label_dict = None
        self.augmentation_strength = 1.0
        os.makedirs(self.dataset_root, exist_ok=True)
        self._load_metadata()

    def _load_metadata(self):
        # Download metadata file
        parquet_path = os.path.join(self.dataset_root, "aquamonitor-jyu.parquet.gzip")
        
        if not os.path.exists(parquet_path):
            hf_hub_download(
                repo_id=self.repo_id,
                filename="aquamonitor-jyu.parquet.gzip",
                repo_type="dataset",
                local_dir=self.dataset_root
            )
            
        metadata = pd.read_parquet(parquet_path)
        
        # Process metadata
        metadata["img"] = metadata["img"].str.removesuffix(".jpg")
        classes = sorted(metadata["taxon_group"].unique())
        self.class_map = {k: v for v, k in enumerate(classes)}
        self.label_dict = dict(zip(metadata["img"], metadata["taxon_group"].map(self.class_map)))

    def get_transforms(self):

        augmentation_strength = self.augmentation_strength

        # Enhanced parameters for underwater conditions
        min_sigma = 0.1
        max_sigma = max(0.1, 5.0 * augmentation_strength)
        water_distortion_scale = 0.3 * augmentation_strength  # Increased from 0.2

        train_transform = transforms.Compose([
            # Base geometric transforms
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # More aggressive cropping
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),

            # Water-specific color distortions
            transforms.ColorJitter(
                brightness=0.4 * augmentation_strength,  # Increased from 0.3
                contrast=0.4 * augmentation_strength,
                saturation=0.4 * augmentation_strength,
                hue=0.15 * augmentation_strength  # Increased from 0.1
            ),
            
            # Underwater motion effects
            transforms.RandomRotation(180 * augmentation_strength),  # Full rotation possible
            transforms.RandomPerspective(
                distortion_scale=water_distortion_scale,
                p=0.5 * augmentation_strength  # Higher probability
            ),
            
            # Depth-related light absorption
            transforms.RandomPosterize(
                bits=int(8 - 4 * augmentation_strength),  # Reduce color depth
                p=0.5 * augmentation_strength
            ),
            
            # Turbidity simulation
            transforms.RandomAutocontrast(
                p=0.3 * augmentation_strength
            ),
            
            # Substrate reflection
            transforms.RandomGrayscale(
                p=0.2 * augmentation_strength  # Simulate silty conditions
            ),

            # Required preprocessing
            transforms.ToTensor(),            
            
            # Water clarity simulation
            transforms.GaussianBlur(
                kernel_size=(5, 9), 
                sigma=(min_sigma, max_sigma)
            ),
            transforms.RandomAdjustSharpness(
                sharpness_factor=2.5 * augmentation_strength,  # Simulate focus variations
                p=0.4 * augmentation_strength
            ),
            
            # Particulate matter simulation
            transforms.RandomErasing(
                p=0.3 * augmentation_strength,
                scale=(0.02, 0.15 * augmentation_strength),  # Streaks/bubbles
                ratio=(0.3, 3.3)  # Vertical/horizontal artifacts
            ),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Validation transforms (unchanged)
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform

    def get_datasets(self):
        train_transform, val_transform = self.get_transforms()

        cache_dir = os.path.join(self.dataset_root, ".cache", "huggingface")
        
        ds = load_dataset(self.repo_id, cache_dir=cache_dir)
        
        def train_preprocess(batch):
            return {
                "key": batch["__key__"],
                "img": [train_transform(x) for x in batch["jpg"]],
                "label": torch.as_tensor([self.label_dict[x] for x in batch["__key__"]], 
                                      dtype=torch.long)
            }
            
        def val_preprocess(batch):
            return {
                "key": batch["__key__"],
                "img": [val_transform(x) for x in batch["jpg"]],
                "label": torch.as_tensor([self.label_dict[x] for x in batch["__key__"]],
                                      dtype=torch.long)
            }

        return (
            ds["train"].with_transform(train_preprocess),
            ds["validation"].with_transform(val_preprocess)
        )
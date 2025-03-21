import os
import pandas as pd
from torchvision import transforms
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch
import numpy as np

class DataHandler:
    def __init__(self, repo_id="mikkoim/aquamonitor-jyu", dataset_root="./dataset"):
        self.repo_id = repo_id
        self.dataset_root = dataset_root
        self.class_map = None
        self.label_dict = None
        self.train_keys = None
        os.makedirs(self.dataset_root, exist_ok=True)
        self._load_metadata()

    def _load_metadata(self):
        parquet_path = os.path.join(self.dataset_root, "aquamonitor-jyu.parquet.gzip")
        
        if not os.path.exists(parquet_path):
            hf_hub_download(
                repo_id=self.repo_id,
                filename="aquamonitor-jyu.parquet.gzip",
                repo_type="dataset",
                local_dir=self.dataset_root
            )
            
        metadata = pd.read_parquet(parquet_path)

        if 'split' not in metadata.columns:
            metadata['split'] = 'train'

        self.train_keys = set(metadata[metadata['split'] == 'train']['img'])
        
        metadata["img"] = metadata["img"].str.removesuffix(".jpg")
        classes = sorted(metadata["taxon_group"].unique())
        self.class_map = {k: v for v, k in enumerate(classes)}
        self.label_dict = dict(zip(metadata["img"], metadata["taxon_group"].map(self.class_map)))
        self.train_keys = set(metadata[metadata['split'] == 'train']['img'])

    def compute_class_weights(self):
        labels = [v for k,v in self.label_dict.items() 
                  if k in self._get_train_keys()] 
        
        class_counts = np.bincount(labels)
        
        class_counts = np.where(class_counts == 0, 1, class_counts)
        
        weights = 1. / class_counts
        
        weights = weights / weights.sum()
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _get_train_keys(self):
        return self.train_keys

    def get_transforms(self):
    
        train_transform = transforms.Compose([
            
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomRotation(15),
            
            transforms.ToTensor(),
            
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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
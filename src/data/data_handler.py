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

    @staticmethod
    def get_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_datasets(self):

         # Configure dataset cache location
        cache_dir = os.path.join(self.dataset_root, ".cache", "huggingface")
        
        # Load dataset with custom cache location
        ds = load_dataset(self.repo_id, cache_dir=cache_dir)
        
        def preprocess(batch):
            return {
                "key": batch["__key__"],
                "img": [self.get_transforms()(x) for x in batch["jpg"]],
                "label": torch.as_tensor([self.label_dict[x] for x in batch["__key__"]], dtype=torch.long)
            }
        
        return (
            ds["train"].with_transform(preprocess),
            ds["validation"].with_transform(preprocess)
        )
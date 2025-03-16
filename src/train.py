import torch
import os
from data.data_handler import DataHandler
from models.resnet import ResNet
from models.model import ConvNeXt
from utils.training_manager import TrainingManager
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def start_resnet_training():    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    output_dir = "outputs_resnet"

    data_handler = DataHandler()
    class_weights = data_handler.compute_class_weights()
    
    ds_train, ds_val = data_handler.get_datasets()
    
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size)
    
    model = ResNet(num_classes=31, class_weights=class_weights)
    
    total_phase1_epochs = 5
    
    optimizer_phase1 = optim.Adam(
        model.parameters(), 
        lr=0.0001,
        weight_decay=model.weight_decay
    )
    
    scheduler_phase1 = ReduceLROnPlateau(
        optimizer_phase1, 
        mode='max',  
        patience=3, 
        factor=0.3, 
        min_lr=1e-6
    )
    
    criterion = torch.nn.CrossEntropyLoss(weight=model.class_weights.to(device))
    trainer = TrainingManager(
        model, 
        optimizer_phase1, 
        criterion, 
        device, 
        output_dir=output_dir,
        scheduler=scheduler_phase1
    )
    trainer.train_phase(train_loader, val_loader, epochs=total_phase1_epochs, phase=1)

    # Phase 2 - Progressive unfreezing
    unfreeze_stages = [(3,4), (5,3), (7,2)]
    
    for stage_idx, (layers, epochs) in enumerate(unfreeze_stages, start=1):
        model.unfreeze_last_layers(layers)
        
        optimizer_phase2 = optim.Adam(
            model.parameters(),
            lr=0.0001,
            weight_decay=model.weight_decay * 0.5
        )

        scheduler_phase2 = ReduceLROnPlateau(
            optimizer_phase2, 
            mode='max',
            patience=2,
            factor=0.5
        )
        
        trainer.update_optimizer(optimizer_phase2, new_scheduler=scheduler_phase2)
        trainer.train_phase(
            train_loader, 
            val_loader, 
            epochs=epochs, 
            phase=2,
            stage=stage_idx
        )

def start_convnext_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    output_dir = "outputs_convnext"

    data_handler = DataHandler()
    class_weights = data_handler.compute_class_weights()
    
    ds_train, ds_val = data_handler.get_datasets()
    
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size)
    
    model = ConvNeXt(num_classes=31, class_weights=class_weights)
    
    total_phase1_epochs = 5
    
    optimizer_phase1 = optim.AdamW(
        model.parameters(),
        lr=0.00005,
        weight_decay=3e-5
    )
    
    scheduler_phase1 = ReduceLROnPlateau(
        optimizer_phase1,
        mode='max',
        patience=4,
        factor=0.25
    )
    
    criterion = torch.nn.CrossEntropyLoss(weight=model.class_weights.to(device))
    trainer = TrainingManager(
        model,
        optimizer_phase1,
        criterion,
        device,
        output_dir=output_dir,
        scheduler=scheduler_phase1
    )
    trainer.train_phase(train_loader, val_loader, epochs=total_phase1_epochs, phase=1)

    unfreeze_stages = [(5,4), (10,3), (15,2)]
    
    for stage_idx, (layers, epochs) in enumerate(unfreeze_stages, start=1):
        model.unfreeze_last_layers(layers)
        
        optimizer_phase2 = optim.AdamW(
            model.parameters(),
            lr=0.0001,
            weight_decay=3e-6
        )

        scheduler_phase2 = ReduceLROnPlateau(
            optimizer_phase2,
            mode='max',
            patience=2,
            factor=0.5
        )
        
        trainer.update_optimizer(optimizer_phase2, new_scheduler=scheduler_phase2)
        trainer.train_phase(
            train_loader,
            val_loader,
            epochs=epochs,
            phase=2,
            stage=stage_idx
        )

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    print("\n=== Training ResNet ===")
    start_resnet_training()
    
    print("\n=== Training ConvNeXt ===")
    start_convnext_training()

if __name__ == "__main__":
    main()
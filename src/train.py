import torch
import os
from data.data_handler import DataHandler
from models.resnet import ResNet
from utils.training_manager import TrainingManager
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def start_training(no_augmentation=False):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    output_dir = "outputs"

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
    
    scheduler_phase1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_phase1, mode='max',  patience=3, factor=0.3, min_lr=1e-6)
    
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

    # Phase 2 - Progressive unfreezing with lower weight decay
    unfreeze_stages = [(3,4), (5,3), (7,2)]
    
    for stage_idx, (layers, epochs) in enumerate(unfreeze_stages, start=1):
        model.unfreeze_last_layers(layers)        
        
        optimizer_phase2 = optim.Adam(
            model.parameters(),
            lr=0.0001,
            weight_decay=model.weight_decay * 0.5
        )

        scheduler_phase2 =  torch.optim.lr_scheduler.ReduceLROnPlateau(
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
            stage=stage_idx,
        )

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    # Run with optimized augmentation
    start_training(no_augmentation=False)

if __name__ == "__main__":
    main()
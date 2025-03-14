import torch
import os
from data.data_handler import DataHandler
from models.resnet import ResNet
from utils.training_manager import TrainingManager
import torch.optim as optim

def start_training(no_augmentation=False, layers_to_freeze=5):    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    output_dir = "outputs"

    data_handler = DataHandler()

    if no_augmentation:
        output_dir = "outputs_no_aug"
        data_handler.augmentation_strength = 0

    if layers_to_freeze == 10:
        output_dir = "outputs_10_layers"

    ds_train, ds_val = data_handler.get_datasets()
    
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size)
    
    model = ResNet(num_classes=31)
    optimizer_phase1 = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = TrainingManager(model, optimizer_phase1, criterion, device, output_dir=output_dir)
    trainer.train_phase(train_loader, val_loader, epochs=2, phase=1)

    model.unfreeze_last_layers(layers_to_freeze)
    optimizer_phase2 = optim.Adam(model.parameters(), lr=0.0001)
    trainer.update_optimizer(optimizer_phase2)
    trainer.train_phase(train_loader, val_loader, epochs=8, phase=2)

def main():
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    start_training(layers_to_freeze=5)

    print("Training (last 5 layers) completed.")

    start_training(layers_to_freeze=10)


if __name__ == "__main__":
    main()
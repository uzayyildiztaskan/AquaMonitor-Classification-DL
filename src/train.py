import torch
from data.data_handler import DataHandler
from models.resnet import ResNet
from models.efficientnet import EfficientNetB3
from utils.training_manager import TrainingManager

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    
    # Data preparation
    data_handler = DataHandler()
    ds_train, ds_val = data_handler.get_datasets()
    
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size)
    
    # Initialize model
    model = ResNet()  # or EfficientNetB3()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training
    trainer = TrainingManager(model, optimizer, criterion, device)
    trainer.train(train_loader, val_loader, epochs=10)

if __name__ == "__main__":
    main()
import torch
import os
from data.data_handler import DataHandler
from models.resnet import ResNet
from utils.training_manager import TrainingManager

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    output_dir = os.path.abspath(output_dir)
    
    data_handler = DataHandler()
    ds_train, ds_val = data_handler.get_datasets()
    
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size)
    
    model = ResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer = TrainingManager(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=output_dir
    )
    
    trainer.train(train_loader, val_loader, epochs=10)

if __name__ == "__main__":
    main()
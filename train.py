import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from dataset import get_dataloaders
from model import MiniInceptionNet

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running: {device}")
    
    os.makedirs("weights", exist_ok=True) 

    writer = SummaryWriter('runs/cifar10_experiment_1')

    train_loader, test_loader, classes = get_dataloaders(batch_size=64)
    model = MiniInceptionNet(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    epochs = 30 
    best_acc = 0.0
    
    patience = 5 
    epochs_no_improve = 0  

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # training
        loop = tqdm(train_loader, leave=True)
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), acc=100.*correct_train/total_train)

        train_acc = 100. * correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)

        #validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        val_acc = 100. * correct_val / total_val
        avg_val_loss = val_loss / len(test_loader)

        scheduler.step()

        print(f"\n=> Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Ghi logs vào TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

  
        if val_acc > best_acc:
            print(f"Val accuracy increased from {best_acc:.2f}% to {val_acc:.2f}%. Saving weights")
            best_acc = val_acc
            torch.save(model.state_dict(), 'weights/best_model.pth')
            epochs_no_improve = 0 
        else:
            epochs_no_improve += 1
            print(f"No improvement. Attempt{epochs_no_improve}/{patience}")
            
            if epochs_no_improve >= patience:
                print(f"Early Stopping! he pattern has stopped improving after {patience} consecutive epochs")
                break 

    writer.close()
    print("Done")

if __name__ == '__main__':
    train_model()
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from model import MiniInceptionNet
from dataset import get_dataloaders

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running: {device}")
    
    _, test_loader, classes = get_dataloaders(batch_size=64)
    
    model = MiniInceptionNet(num_classes=10).to(device)
    try:
        # weights_only=True 
        model.load_state_dict(torch.load('weights/best_model.pth', map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Errors: Not found 'weights/best_model.pth'.")
        return

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(all_targets, all_preds, target_names=classes))

    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix - Custom Mini InceptionNet")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\nSave Confusion_matrix as 'confusion_matrix.png'.")
    plt.show()

if __name__ == '__main__':
    evaluate()
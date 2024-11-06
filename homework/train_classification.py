import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.inference_mode():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(val_loader)
    return avg_loss, accuracy

def main():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = load_data(
        'classification_data/train',
        return_dataloader=True,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        transform_pipeline="aug"
    )
    
    val_loader = load_data(
        'classification_data/val',
        return_dataloader=True,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )
    
    num_epochs = 10
    best_acc = 0
    patience = 3
    no_improve = 0
    
    print(f"Starting training on device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    try:
        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            scheduler.step(val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
                print(f"  Saving best model with accuracy: {best_acc:.2f}%")
                save_model(model)
            else:
                no_improve += 1
                
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs. Stopping training.")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if best_acc > 0:
            print("Saving last best model...")
            save_model(model)
    
    print("Training finished!")
    print(f"Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
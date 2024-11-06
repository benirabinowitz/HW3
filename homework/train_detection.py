import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.models import Detector, save_model
from homework.metrics import DetectionMetric
from homework.datasets.road_dataset import load_data

def train(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_depth_loss = 0.0
    
    for batch in train_loader:
        images = torch.tensor(batch["image"], dtype=torch.float32).to(device, non_blocking=True)
        depths = torch.tensor(batch["depth"], dtype=torch.float32).to(device, non_blocking=True)
        labels = torch.tensor(batch["track"], dtype=torch.long).to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        logits, pred_depths = model(images)
        
        seg_loss = nn.CrossEntropyLoss()(logits, labels)
        depth_loss = nn.MSELoss()(pred_depths, depths)
        total_loss = seg_loss + 0.5 * depth_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += total_loss.item()
        running_seg_loss += seg_loss.item()
        running_depth_loss += depth_loss.item()
    
    n = len(train_loader)
    return running_loss/n, running_seg_loss/n, running_depth_loss/n

def validate(model, val_loader, metric, device):
    model.eval()
    metric.reset()
    
    with torch.inference_mode():
        for batch in val_loader:
            images = torch.tensor(batch["image"], dtype=torch.float32).to(device, non_blocking=True)
            depths = torch.tensor(batch["depth"], dtype=torch.float32).to(device, non_blocking=True)
            labels = torch.tensor(batch["track"], dtype=torch.long).to(device, non_blocking=True)
            
            logits, pred_depths = model(images)
            pred_labels = logits.argmax(dim=1)
            
            metric.add(pred_labels, labels, pred_depths, depths)
    
    return metric.compute()

def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = load_data(
        'road_data/train',
        return_dataloader=True,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = load_data(
        'road_data/val',
        return_dataloader=True,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )
    
    model = Detector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.7,  
        patience=2,  
        min_lr=1e-5  
    )
    metric = DetectionMetric()
    
    num_epochs = 20
    best_iou = 0.0
    patience = 5
    no_improve = 0

    print(f"Starting training on device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    try:
        for epoch in range(num_epochs):
            train_loss, seg_loss, depth_loss = train(model, train_loader, optimizer, device)
            
            metrics = validate(model, val_loader, metric, device)
            current_iou = metrics["iou"]
            
            scheduler.step(current_iou)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f} (Seg: {seg_loss:.4f}, Depth: {depth_loss:.4f})')
            print(f'  Val IoU: {metrics["iou"]:.4f}')
            print(f'  Val Abs Depth Error: {metrics["abs_depth_error"]:.4f}')
            print(f'  Val TP Depth Error: {metrics["tp_depth_error"]:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

            if current_iou > best_iou:
                best_iou = current_iou
                no_improve = 0
                print(f"  Saving best model with IoU: {best_iou:.4f}")
                save_model(model)
            else:
                no_improve += 1
                
            if no_improve >= patience:
                print(f"No improvement for {patience} epochs. Stopping training.")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if best_iou > 0:
            print("Saving last best model")
            save_model(model)

    print("Training finished!")
    print(f"Best IoU: {best_iou:.4f}")

if __name__ == '__main__':
    main()
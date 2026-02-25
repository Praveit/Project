import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Enable optimizations for RTX cards
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def prepare_data(batch_size=8192):
    data = np.loadtxt('cardio_train.csv', delimiter=',', skiprows=1)
    X = data[:, 1:-1]
    y = data[:, -1]

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    return train_loader, X_test, y_test


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, criterion, optimizer, scheduler, train_loader, X_test, y_test, epochs=100000):
    best_loss = float('inf')
    patience_counter = 0
    patience = 500
    scaler = torch.amp.GradScaler('cuda')
    
    print("Starting training loop...")
    print(f"Model is on: {next(model.parameters()).device}")
    print(f"Test data is on: {X_test.device}")
    
    # Warm up GPU
    for _ in range(3):
        with torch.amp.autocast('cuda'):
            _ = model(torch.randn(batch_size, X_test.shape[1], device=device))
    torch.cuda.synchronize()
    
    import time
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"Epoch {epoch}: LR reduced {old_lr:.6f} → {new_lr:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad(), torch.amp.autocast('cuda'):
                test_outputs = model(X_test)
                test_preds = (torch.sigmoid(test_outputs) >= 0.5).float()
                test_acc = (test_preds == y_test).sum().item() / y_test.size(0) * 100
            
            elapsed = time.time() - start_time
            epochs_per_sec = (epoch + 1) / elapsed if elapsed > 0 else 0
            
            print(f"Epoch {epoch:5d} | Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | "
                  f"Test: {test_acc:.2f}% | LR: {new_lr:.6f} | Speed: {epochs_per_sec:.2f} epochs/sec")
            
            # Check GPU utilization
            if epoch == 0:
                print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# Main
batch_size = 8192
train_loader, X_test, y_test = prepare_data(batch_size=batch_size)

first_batch = next(iter(train_loader))
input_size = first_batch[0].shape[1]

model = NeuralNetwork(input_size).to(device)

print(f"\nModel architecture on {next(model.parameters()).device}:")
print(model)
print(f"\nRunning with mixed precision")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-7
)

print(f"Batch size: {batch_size}")
print(f"Batches per epoch: {len(train_loader)}\n")

train_model(model, criterion, optimizer, scheduler, train_loader, X_test, y_test, epochs=100000)

torch.save(model.state_dict(), "model.pth")
print("\nModel saved!")
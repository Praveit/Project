import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")


# Data prep
def prepare_data(batch_size=32):
    data = np.loadtxt('cardio_train.csv', delimiter=',', skiprows=1)
    X = data[:, 1:-1]  # Exclude the first column (ID) and last column (target)
    y = data[:, -1]    # Target column

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Normalize the features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors and move to GPU
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, X_train, X_test, y_train, y_test


# Define/Create model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Training with batches
def train_model(model, criterion, optimizer, scheduler, train_loader, X_train, y_train, epochs=1000):
    best_loss = float('inf')
    patience_counter = 0
    patience = 500
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Batch training
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Get current learning rate before scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step the scheduler
        old_lr = current_lr
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Manually print LR changes
        if old_lr != new_lr:
            print(f"Epoch {epoch}: Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 100 == 0:
            # Calculate accuracy on full training set
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train)
                train_preds = (train_outputs >= 0.5).float()
                train_acc = (train_preds.eq(y_train).sum().item()) / y_train.size(0)
            
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Train Acc: {train_acc*100:.2f}%, LR: {new_lr:.6f}")


# Evaluating model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = (predictions >= 0.5).float()
        accuracy = (predictions.eq(y_test).sum().item()) / y_test.size(0)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# Main
batch_size = 8192  # Adjust based on your GPU memory
train_loader, X_train, X_test, y_train, y_test = prepare_data(batch_size=batch_size)

input_size = X_train.shape[1]
model = NeuralNetwork(input_size).to(device)  # Move model to GPU

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Add learning rate scheduler (removed verbose parameter)
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=100, 
    min_lr=1e-7
)

train_model(model, criterion, optimizer, scheduler, train_loader, X_train, y_train, epochs=100000)

# Evaluate
evaluate_model(model, X_test, y_test)

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved")
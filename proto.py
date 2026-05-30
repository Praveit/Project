import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

# Data prep
def prepare_data():
    # Load the new, cleaned dataset
    df = pd.read_csv('cardio_train_cleaned.csv')
    data = df.to_numpy()
    
    X = data[:, 1:-1]  # Exclude first column (ID) and last column (target)
    y = data[:, -1]    # Target column

    # Split BEFORE normalizing to prevent data leakage!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calculate mean and std ONLY from the training set
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Apply normalization to both sets
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test

# Define/Create model with regularization
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)         # Batch Normalization added
        self.dropout1 = nn.Dropout(0.2)       # Dropout added to prevent overfitting
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Training Loop with Batches and Scheduler
def train_model(model, criterion, optimizer, scheduler, train_loader, X_test, y_test, epochs=150):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Iterate over batches instead of the whole dataset
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Evaluate validation loss to step the scheduler
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
        
        scheduler.step(val_loss) # Drops learning rate if val_loss plateaus
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f} | LR: {current_lr:.6f}")


# Main Execution
X_train, X_test, y_train, y_test = prepare_data()

# Wrap data in a DataLoader for mini-batching
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

input_size = X_train.shape[1]
model = NeuralNetwork(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

print("Starting training...")
train_model(model, criterion, optimizer, scheduler, train_loader, X_test, y_test, epochs=150)

# Evaluating model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = (predictions >= 0.5).float()
        accuracy = (predictions.eq(y_test).sum().item()) / y_test.size(0)
        print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    
evaluate_model(model, X_test, y_test)

torch.save(model.state_dict(), "model.pth")
print("Model Saved.")
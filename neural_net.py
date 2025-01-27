import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),  
            nn.Linear(512, 512),
            nn.ReLU(), 
            nn.Linear(512, 512),
            nn.ReLU(),  
            nn.Linear(512, 512),  
            nn.ReLU(),  
            nn.Linear(512, 512),  
            nn.ReLU(),  
            nn.Linear(512, num_classes)  # Output layer, no activation (logits)
        )

    def forward(self, x):
        return self.layers(x)
    
def train_nn(df):
    # Shuffle and split the data into training, validation, and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Prepare training data
    X_train = train_df.drop(columns=["player1_won"]).values
    Y_train = train_df["player1_won"].values
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Prepare validation data
    X_val = val_df.drop(columns=["player1_won"]).values
    Y_val = val_df["player1_won"].values
    x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Prepare test data
    X_test = test_df.drop(columns=["player1_won"]).values
    Y_test = test_df["player1_won"].values
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    # Initialize the model, loss function, optimizer, and scheduler
    model = NeuralNetwork(input_size=len(df.columns) - 1, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    num_epochs = 20
    for epoch in range(num_epochs):
        # Training step
        model.train()
        train_loss = 0
        for batch_features, batch_labels in train_dataloader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_dataloader)  # Average training loss for the epoch

        # Validation step (compute validation loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_features, val_labels in val_dataloader:
                val_outputs = model(val_features)
                val_loss += criterion(val_outputs, val_labels).item()
        val_loss /= len(val_dataloader)  # Average validation loss for the epoch

        # Adjust learning rate based on validation loss
        
        scheduler.step(val_loss)
        print("New learning rate: ", scheduler.get_last_lr())
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        outputs = model(x_test_tensor)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test_tensor).sum().item()
        total = y_test_tensor.size(0)
        accuracy = correct / total

    print(f"Test Accuracy: {accuracy:.4f}")
    
def main():
    df = joblib.load("processed_df.pkl")   
    train_nn(df)
if __name__ == "__main__":
    main()
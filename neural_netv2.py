from typing import OrderedDict
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import shap  # SHAP for feature importance
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def create_layers(num_layers, num_params_per_layer, input_size, num_classes):
    result = OrderedDict()
    result["layer0"] = nn.Linear(input_size, num_params_per_layer)
    result["relu0"] = nn.ReLU()
    for i in range(1, num_layers + 1):
        result[f"layer{i}"] = nn.Linear(num_params_per_layer, num_params_per_layer)
        result[f"relu{i}"] = nn.ReLU()
    result["output"] = nn.Linear(num_params_per_layer, num_classes)
    return result

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        num_params_per_layer = 512
        self.layers = nn.Sequential(create_layers(4, num_params_per_layer, input_size, num_classes))

    def forward(self, x):
        return self.layers(x)
def normalize_features(df, y_column):
    """ Standardizes features to prevent bias from large values. """
    feature_columns = [col for col in df.columns if col != y_column]
    
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])  # Normalize all input features

    return df, scaler
def train_nn(df):
    y_column = "PlayerTeam1.won"
    
    # Normalize Features
    #df, scaler = normalize_features(df, y_column)
    # Shuffle and split the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Prepare datasets
    X_train = train_df.drop(columns=[y_column]).values
    Y_train = train_df[y_column].values
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    X_val = val_df.drop(columns=[y_column]).values
    Y_val = val_df[y_column].values
    x_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    X_test = test_df.drop(columns=[y_column]).values
    Y_test = test_df[y_column].values
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    # Initialize model, loss, and optimizer
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size=input_size, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_features, batch_labels in train_dataloader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_labels).sum().item()
            train_total += batch_labels.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for val_features, val_labels in val_dataloader:
                val_outputs = model(val_features)
                val_loss += criterion(val_outputs, val_labels).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == val_labels).sum().item()
                val_total += val_labels.size(0)
        
        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Evaluate model on test set
    model.eval()
    with torch.no_grad():
        outputs = model(x_test_tensor)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test_tensor).sum().item()
        total = y_test_tensor.size(0)
        accuracy = correct / total

    print(f"Test Accuracy: {accuracy:.4f}")

    # **SHAP Feature Importance Analysis**
    #compute_shap_values(model, df.drop(columns=[y_column]))

def compute_shap_values(model, X_df):
    """ Compute SHAP values for feature importance """
    model.eval()  # Set model to evaluation mode

    # Convert DataFrame to Tensor
    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)

    # Define a wrapper function for SHAP
    def model_wrapper(x_numpy):
        """ SHAP expects numpy inputs; convert to tensor, get predictions, convert back """
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x_tensor)  # Get raw output
            probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]  # Get probability for class 1
        return probs.numpy()

    # Define SHAP explainer
    explainer = shap.Explainer(model_wrapper, X_df.to_numpy())
    shap_values = explainer(X_df.to_numpy())

    # Plot SHAP summary (shows overall feature importance)
    feature_names = list(X_df.columns)
    shap.summary_plot(shap_values, X_df, feature_names=feature_names)

    print("SHAP values calculated. Check the plot.")
def main():
    df = joblib.load("./data/preprocessed_df.pkl")
        
    
    print("Total missing values after cleanup:", df.isna().sum().sum())
    train_nn(df)

if __name__ == "__main__":
    main()

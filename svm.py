import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from libsvm.svmutil import *
import joblib

def convert_df_to_libsvm(df, target_column):
    """
    Converts a Pandas DataFrame to LIBSVM format.
    
    Args:
        df (pd.DataFrame): The dataset.
        target_column (str): The name of the column containing labels.
    
    Returns:
        Y (list): Labels (-1 or 1).
        X (list of dicts): Features in sparse format {index: value}.
    """
    # Extract features and labels

    Y = df[target_column].values
    X = df.drop(columns=[target_column]).values  # Drop target column to get features

    # Normalize features to 0-1 range (important for SVM performance)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to LIBSVM sparse format
    X_libsvm = [{i + 1: X_scaled[j][i] for i in range(len(X_scaled[j]))} for j in range(len(X_scaled))]

    # Convert labels: LIBSVM prefers -1 and 1 for binary classification
    Y = np.where(Y == 0, -1, 1).tolist()
    print(len(Y))

    return Y, X_libsvm, scaler

def train_svm(Y, X):
    """
    Trains an SVM model using LIBSVM.

    Args:
        Y (list): Labels (-1 or 1).
        X (list of dicts): Features in LIBSVM format.

    Returns:
        model: Trained LIBSVM model.
    """
    # Create SVM problem
    problem = svm_problem(Y, X)

    # Set SVM parameters (RBF Kernel, C=1, gamma=0.5)
    param = svm_parameter('-t 2 -c 1 -g 0.5')

    # Train the model
    model = svm_train(problem, param)
    return model

def evaluate_svm(model, df_test, target_column, scaler):
    """
    Evaluates the SVM model on a test DataFrame.

    Args:
        model: Trained LIBSVM model.
        df_test (pd.DataFrame): Test set.
        target_column (str): Column name of labels.
        scaler: Fitted MinMaxScaler (from training set).

    Returns:
        accuracy (float): Test accuracy.
    """
    # Extract labels and features
    Y_test = df_test[target_column].values
    X_test = df_test.drop(columns=[target_column]).values

    # Normalize test features using the training scaler
    X_test_scaled = scaler.transform(X_test)

    # Convert to LIBSVM sparse format
    X_test_libsvm = [{i + 1: X_test_scaled[j][i] for i in range(len(X_test_scaled[j]))} for j in range(len(X_test_scaled))]

    # Convert labels
    Y_test = np.where(Y_test == 0, -1, 1).tolist()

    # Predict using the trained model
    predicted_labels, accuracy, _ = svm_predict(Y_test, X_test_libsvm, model)

    print(f"Test Accuracy: {accuracy[0]:.2f}%")
    return accuracy[0]

def main():
    target_column = "player1_won"
    df = joblib.load("processed_df.pkl")
    df = df.sample(frac=1, random_state=42)
    print(len(df))
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    # Convert DataFrame to LIBSVM format
    Y_train, X_train, scaler = convert_df_to_libsvm(train_df, target_column)
    Y_test, X_test, _ = convert_df_to_libsvm(test_df, target_column)

    # Train the SVM model
    svm_model = train_svm(Y_train, X_train)

    # Evaluate on test set
    evaluate_svm(svm_model, test_df, target_column, scaler)
    
if __name__ == "__main__":
    main()
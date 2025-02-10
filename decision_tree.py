import xgboost as xgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv

from sklearn.model_selection import HalvingGridSearchCV
import joblib
print("Scikit-Learn Version:", sklearn.__version__)
print("XGBoost Version:", xgb.__version__)
def search(X_train, X_test, Y_train, Y_test):
    # Define the XGBoost model
    xgb_model = xgb.XGBClassifier(eval_metric="logloss")

    # Hyperparameter grid for tuning
    param_grid = {
    "max_depth": [None],                   # Depth of trees
    "subsample": [0.8],                 # Fraction of data to use per tree
    "colsample_bytree": [0.8],          # Fraction of features per tree
    "lambda": [1],                      # L2 regularization
    "alpha": [0],  
    "eta": [.01],                          # L1 regularization
    "n_estimators": [2500,2750, 3000, 3250, 3500]
}

    # Perform HalvingGridSearchCV using all available cores (n_jobs=-1)
    halving_search = HalvingGridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="accuracy",
        factor=2,  # Reduce candidates by a factor of 3 each iteration
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42  # For reproducibility
    )
    
    # Fit the model
    halving_search.fit(X_train, Y_train)

    # Output the best hyperparameters
    print("Best Hyperparameters:", halving_search.best_params_)

    return halving_search.best_estimator_
    
def train(X_train, X_test, Y_train, Y_test):
    # Convert to XGBoost DMatrix (optimized format for speed)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    # Define hyperparameters
    params = {
        "objective": "binary:logistic",  # Binary classification
        "eval_metric": "logloss",        # Loss function
        "eta": 0.01,                      # Learning rate
        "max_depth": None,                   # Depth of trees
        "subsample": 0.8,                 # Fraction of data to use per tree
        "colsample_bytree": 0.8,          # Fraction of features per tree
        "lambda": 1,                      # L2 regularization
        "alpha": 0,                       # L1 regularization
    }
    # Train the model
    num_rounds = 2750  # Number of boosting rounds
    bst = xgb.train(params, dtrain, num_rounds)
    # Predict probabilities
    y_pred_proba = bst.predict(dtest)

    # Convert to binary (0 or 1)
    y_pred = np.where(y_pred_proba > 0.5, 1, 0)

    # Evaluate accuracy
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    bst.save_model("xgboost_model.json")

def main():
    # Load dataset
    df = joblib.load("./data/preprocessed_df.pkl")
    df = df.sample(frac=1, random_state=42)
    # Define target and features
    target_column = "PlayerTeam1.won"
    X = df.drop(columns=[target_column]).values
    Y = df[target_column].values 

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # Normalize features (XGBoost can handle raw data, but scaling helps)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    train(X_train, X_test, Y_train, Y_test)
    #search(X_train, X_test, Y_train, Y_test)
if __name__ == "__main__":
    main()

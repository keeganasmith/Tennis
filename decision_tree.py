import xgboost as xgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
print("Scikit-Learn Version:", sklearn.__version__)
print("XGBoost Version:", xgb.__version__)
def search(X_train, X_test, Y_train, Y_test):
    # Define the XGBoost model (note: setting use_label_encoder=False avoids a warning)
    xgb_model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)

    # Expanded hyperparameter grid for tuning
    param_grid = {
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "n_estimators": [100, 200, 300],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [0, 1, 10],   # L2 regularization
        "reg_alpha": [0, 0.1, 1]    # L1 regularization
    }

    # Perform grid search using all available cores (n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, Y_train)

    # Output the best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)
    
def train(X_train, X_test, Y_train, Y_test):
    # Convert to XGBoost DMatrix (optimized format for speed)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    # Define hyperparameters
    params = {
        "objective": "binary:logistic",  # Binary classification
        "eval_metric": "logloss",        # Loss function
        "eta": 0.1,                      # Learning rate
        "max_depth": 6,                   # Depth of trees
        "subsample": 0.8,                 # Fraction of data to use per tree
        "colsample_bytree": 0.8,          # Fraction of features per tree
        "lambda": 1,                      # L2 regularization
        "alpha": 0,                       # L1 regularization
    }
    # Train the model
    num_rounds = 200  # Number of boosting rounds
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

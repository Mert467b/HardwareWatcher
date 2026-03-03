import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models import (
    DecisionTreeRegression, 
    RandomForestRegressor, 
    DecisionTreeClassification, 
    RandomForestClassifier

)
from src.utils import (
    calculate_mape,
    evaluate_model,
    get_feature_importance,
    preprocess_regression_data,
    evaluate_overfitting,
    
    preprocess_classification_data,
    classification_accuracy,
    confusion_matrix,
    calculate_metrics,
    calculate_gap
)
from src.visualization import (
    scatter_plot, 
    histogram,
    residual_plot
)

TASK = 'classification' # Options: 'regression' or 'classification'
RUN_PLOTS = True    # True for plots on False for off
DATA_FILE = 'data_logs.csv'


def run_pipeline(data_path):
    df = pd.read_csv(data_path)
    X, y, features = preprocess_regression_data(df)

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    if TASK == 'regression':
        X, y, features = preprocess_regression_data(df)
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        train_idx, test_idx = indices[:split], indices[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        model = RandomForestRegressor(n_trees=15)
    
    else:
        X, y, features = preprocess_classification_data(df)
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(X))
        train_idx, test_idx = indices[:split], indices[split:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        model = RandomForestClassifier(n_trees=15)
    
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    train_preds = model.predict(X_train)
    y_pred = test_preds
    y_true = y_test
    
    if TASK == 'regression':
        
        mape_score = calculate_mape(y_test, test_preds)
        print(f"MAPE: {mape_score:.2f}%")
        print(f"Model Accuracy: {100 - mape_score:.2f}%")

        mae, r2, rmse = evaluate_model(y_test, test_preds)
        print("\n--- Model Evaluation Results ---")
        print(f"Mean Absolute Error (MAE):   {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE):    {rmse:.2f}")
        print(f"R-squared (R2):              {r2:.4f}")
        print("--------------------------------")

        importances = get_feature_importance(model, features)
        for feat, val in importances.items():
            print(f"{feat}: {val:.4f}")
    
        train_r2, test_r2, train_mse, test_mse = evaluate_overfitting(model, X_train, y_train, X_test, y_test)
        print(f"\n--- Model Performance ---")
        print(f"Train R2 Score: {train_r2:.4f}")
        print(f"Test R2 Score:  {test_r2:.4f}")
        print(f"Train MSE:      {train_mse:.4f}")
        print(f"Test MSE:       {test_mse:.4f}")

        gap = train_r2 - test_r2
        if gap > 0.15:
            print(f"\n[!] WARNING: High Overfitting detected. Gap: {gap:.4f}")
        elif train_r2 > 0.99 and test_r2 > 0.99:
            print(f"\n[!] WARNING: Potential Data Leakage. Scores are suspiciously high.")
        else:
            print("\n[+] Model generalization looks reasonable.")
        
        if RUN_PLOTS:
            scatter_plot(y_test, test_preds, title='Random Forest: Actual vs Predicted')
            residual_plot(y_test, test_preds, title = 'Residual Plot: CPU Temperature Prediction')
            histogram(y_test, test_preds, title = 'Histogram of Prediction Errors')
    
    elif TASK == 'classification':
        accuracy = classification_accuracy(model, X_test, y_test)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        
        p0, r0, f1_0, s0, p1, r1, f1_1, s1, acc = confusion_matrix(y_true, y_pred)
        print("Classification Report:")
        print(f"{'':>15} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}")
        print(f"\n{'0.0':>15} {p0:>9.2f} {r0:>9.2f} {f1_0:>9.2f} {s0:>9}")
        print(f"{'1.0':>15} {p1:>9.2f} {r1:>9.2f} {f1_1:>9.2f} {s1:>9}")
        print(f"\n{'accuracy':>15} {'':>9} {'':>9} {acc:>9.2f} {len(y_true):>9}")
        
        precision, recall, f1 = calculate_metrics(y_true, y_pred)
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")

        train_acc, test_acc = calculate_gap(model, X_train, X_test, y_train, y_test)
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy:  {test_acc:.4f}")
        print(f"Gap: {train_acc - test_acc:.4f}")
        if RUN_PLOTS:
            pass

if __name__ == "__main__":
    run_pipeline(DATA_FILE)
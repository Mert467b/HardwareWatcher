import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import (
    DecisionTreeClassification, 
    RandomForestClassifier
)
from src.utils import (
    preprocess_classification_data,
    classification_accuracy,
    print_confusion_matrix,
    calculate_metrics,
    calculate_gap
)
from src.visualization import (
    plot_confusion_matrix, 
    test_temporal_degradation, 
    test_walk_forward)
from xgboost import XGBClassifier

TASK = 'RandomForest' # Options: 'RandomForest' or 'XGBoost'
RUN_PLOTS = True    # True for plots on False for off
DATA_FILE = 'data_logs.csv'


def run_pipeline(data_path):
    df = pd.read_csv(data_path)
    X, y, features, prediction_window, df_processed = preprocess_classification_data(df)
    
    if TASK == 'RandomForest':
        np.random.seed(42)
        split_idx = int(len(df_processed) * 0.8)
 
        train_df = df_processed.iloc[:split_idx - prediction_window]
        test_df  = df_processed.iloc[split_idx:]
 
        X_train, y_train = train_df[features].values, train_df["target_future_throttle"].values
        X_test,  y_test  = test_df[features].values,  test_df["target_future_throttle"].values
        model = RandomForestClassifier(n_trees=50, max_depth=4, min_samples_leaf=50, min_samples_split=100)
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)
        train_preds = model.predict(X_train)
        y_pred = test_preds
        y_true = y_test

    else:
        np.random.seed(42)
        split_idx = int(len(df_processed) * 0.8)
 
        train_df = df_processed.iloc[:split_idx - prediction_window]
        test_df  = df_processed.iloc[split_idx:]
 
        X_train, y_train = train_df[features].values, train_df["target_future_throttle"].values
        X_test,  y_test  = test_df[features].values,  test_df["target_future_throttle"].values

        xgb_model = XGBClassifier(
            n_estimators=150,        # Slightly more trees to compensate for slower learning
            max_depth=3,             # Shallower trees to prevent memorization
            learning_rate=0.1,      # Slower, more cautious learning
            subsample=0.8,           # Random Forest-style row sampling
            colsample_bytree=0.8,    # Random Forest-style feature sampling
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
    
    
    if TASK == 'RandomForest':
        accuracy = classification_accuracy(model, X_test, y_test)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
        p0, r0, f1_0, s0, p1, r1, f1_1, s1, acc = print_confusion_matrix(y_true, y_pred)
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
            _, _, fig, ax = plot_confusion_matrix(y_true, y_pred)
            plt.show()
            
            test_temporal_degradation(model,df_processed, features, prediction_window)
            
            test_walk_forward(model, df_processed, features, prediction_window)
    
    elif TASK == 'XGBoost':
        y_probs = xgb_model.predict_proba(X_test)[:, 1]

        custom_threshold = 0.85 

        y_pred = (y_probs >= custom_threshold).astype(int)
        y_true = y_test

        accuracy = np.mean(y_pred == y_true)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

        p0, r0, f1_0, s0, p1, r1, f1_1, s1, acc = print_confusion_matrix(y_true, y_pred)
        print("Classification Report:")
        print(f"{'':>15} {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9}")
        print(f"\n{'0.0':>15} {p0:>9.2f} {r0:>9.2f} {f1_0:>9.2f} {s0:>9}")
        print(f"{'1.0':>15} {p1:>9.2f} {r1:>9.2f} {f1_1:>9.2f} {s1:>9}\n")

        precision, recall, f1 = calculate_metrics(y_true, y_pred)
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}\n")

        train_acc, test_acc = calculate_gap(xgb_model, X_train, X_test, y_train, y_test)
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy:  {test_acc:.4f}")
        print(f"Gap: {train_acc - test_acc:.4f}")

if __name__ == "__main__":
    run_pipeline(DATA_FILE)
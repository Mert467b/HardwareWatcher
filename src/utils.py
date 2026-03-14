import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

def preprocess_classification_data(df):
    print(f"1. Original Rows: {len(df)}")
 
    for col in ["activity", "surface_type", "cpu_boost_mode"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
 
    activity_map = {"idle": 0, "light_load": 1, "medium_load": 2, "heavy_load": 3, "cooling": 4}
    df["activity"]       = df["activity"].map(activity_map)
    df["surface_type"]   = df["surface_type"].map({"soft": 0, "rough": 1})
    df["cpu_boost_mode"] = df["cpu_boost_mode"].map({"disabled": 0, "aggressive": 1})
 
    for col in ["activity", "surface_type", "cpu_boost_mode"]:
        if df[col].isnull().any():
            print(f"Warning: Unexpected values found in {col}. Check your CSV for typos.")
    
    for col in ["cpu_temp_C", "cpu_power_W", "cpu_util_pct"]:
        df[f"{col}_roll_mean_30"] = df[col].rolling(30).mean()
        df[f"{col}_roll_std_30"]  = df[col].rolling(30).std()
        df[f"{col}_roll_mean_60"] = df[col].rolling(60).mean()
    
    gpu_cols_available = []
    for col in ["gpu_util_pct", "gpu_temp_C", "gpu_power_W", "gpu_clock_MHz"]:
        if col in df.columns:
            gpu_cols_available.append(col)
        else:
            print(f"Warning: {col} not found in CSV — skipping.")
 
    print(f"  GPU columns found: {gpu_cols_available}")
 
    df = df.ffill()
    df = df.dropna()
 
    df["is_throttling_now"] = ((df["cpu_temp_C"] > 85) & (df["cpu_freq_MHz"] < 2000)).astype(int)
 
    prediction_window = 30
 
    df["target_future_throttle"] = (
        df["is_throttling_now"]
        .shift(-prediction_window)
        .rolling(window=prediction_window, min_periods=1)
        .max()
        
    )
 
    to_lag = ["cpu_power_W", "cpu_util_pct", "cpu_temp_C"] + gpu_cols_available
    for col in to_lag:
        for lag in [5, 10, 30]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
 
    if "cpu_temp_slope" in df.columns:
        df["cpu_slope_lag_5"] = df["cpu_temp_slope"].shift(5)
    else:
        df["cpu_temp_slope"]  = df["cpu_temp_C"].diff()
        df["cpu_slope_lag_5"] = df["cpu_temp_slope"].shift(5)
 
    if "gpu_temp_C" in gpu_cols_available:
        df["gpu_temp_slope"]   = df["gpu_temp_C"].diff()
        df["gpu_slope_lag_5"]  = df["gpu_temp_slope"].shift(5)
 
    df["freq_headroom"] = df["cpu_freq_MHz"] / 2000
 
    if "gpu_clock_MHz" in gpu_cols_available:
        gpu_max_freq = df["gpu_clock_MHz"].max()
        df["gpu_freq_headroom"] = df["gpu_clock_MHz"] / gpu_max_freq
        print(f"  GPU max freq used for headroom: {gpu_max_freq:.0f} MHz")
 
    df = df.dropna().reset_index(drop=True)
    print(f"2. Rows after dropping shift-induced NaNs: {len(df)}")
 
    features = [
    # Categorical
        "activity", "surface_type", "cpu_boost_mode",
 
    # Current State (Time t) - Crucial for baseline!
        "cpu_power_W",
        "cpu_util_pct",
        "cpu_temp_C",
        
        "cpu_temp_C_roll_mean_30", "cpu_temp_C_roll_std_30",
        "cpu_temp_C_roll_mean_60",
        "cpu_power_W_roll_mean_30", "cpu_power_W_roll_std_30",
        
    # GPU Current State (if available)
        "gpu_temp_C",
 
    # CPU lag features (Time t-5, t-10)
        "cpu_power_W_lag_5", "cpu_power_W_lag_10",
        "cpu_util_pct_lag_5",  "cpu_util_pct_lag_10",
        "cpu_temp_C_lag_5",    "cpu_temp_C_lag_10",
 
    # CPU slope and headroom
        "cpu_slope_lag_5", "freq_headroom",
        
    # GPU lag features
        "gpu_temp_C_lag_5", "gpu_temp_C_lag_10"]
 
 
    features = [f for f in features if f in df.columns]
    print(f"\n  Total features: {len(features)}")
    print(f"  → {features}")
    
    X = df[features].values
    y = df["target_future_throttle"].values
    return X, y, features, prediction_window, df

def classification_accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    return accuracy

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def calculate_gap(model, X_train, X_test, y_train, y_test):
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = np.mean(train_preds == y_train)
    test_acc = np.mean(test_preds == y_test)
    return train_acc, test_acc

def print_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    print("Confusion Matrix:")
    print(f"[[{tn}  {fp}]\n [{fn}  {tp}]]\n")

    def get_stats(true, pred, label):
        subset_true = (true == label)
        tp_s = np.sum((true == label) & (pred == label))
        fp_s = np.sum((true != label) & (pred == label))
        fn_s = np.sum((true == label) & (pred != label))
        support = np.sum(subset_true)
        
        prec = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
        rec = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        return prec, rec, f1, support

    p0, r0, f1_0, s0 = get_stats(y_true, y_pred, 0)
    p1, r1, f1_1, s1 = get_stats(y_true, y_pred, 1)
    acc = np.mean(y_true == y_pred)
    return p0, r0, f1_0, s0, p1, r1, f1_1, s1, acc 



def quick_fit(model, X_tr, y_tr, X_te, y_te):
    
    model.fit(X_tr, y_tr)
    
    preds = model.predict(X_te)
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
    else:
        auc = 0.0 
        
    acc = accuracy_score(y_te, preds)
    
    return model, acc, auc






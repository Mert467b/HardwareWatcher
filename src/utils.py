import numpy as np
import pandas as pd

def preprocess_regression_data(df):
    df["activity"] = df["activity"].map({"idle": 0, "light_load":1, "medium_load":2 })
    df["surface_type"] = df["surface_type"].map({"soft":0, "rough":1})
    df["cpu_boost_mode"] = df["cpu_boost_mode"].map({"disabled":0, "aggressive":1})

    features = ["activity",	"surface_type",	"cpu_boost_mode",
            "cpu_temp_slope", "cpu_power_W","cpu_util_pct",	
            "cpu_freq_MHz", "ram_used_GB", "gpu_temp_C", 
            "gpu_temp_slope", "gpu_util_pct", "gpu_clock_MHz","gpu_power_W"]

    X = df[features].values
    y = df["cpu_temp_C"].values

    return X, y, features

def calculate_mape(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return mape

def evaluate_model(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = (np.mean((y_true - y_pred)**2))
    rmse = np.sqrt(mse)
    
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mae, r2, rmse

def get_feature_importance(model, feature_names):
    importance = {name: 0 for name in feature_names}
    
    def traverse_tree(node):
        if not isinstance(node, dict): return
        feat_name = feature_names[node['feature']]
        importance[feat_name] += 1
        traverse_tree(node['left'])
        traverse_tree(node['right'])
        
    for tree in model.trees:
        traverse_tree(tree.tree)
        
    total = sum(importance.values())
    return {k: v / total for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}

def evaluate_overfitting(model, X_train, y_pred, X_test, y_true):
    import numpy as np

def evaluate_overfitting(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    def calculate_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)


    def calculate_mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    train_r2 = calculate_r2(y_train, train_pred)
    train_mse = calculate_mse(y_train, train_pred)
    
    
    test_r2 = calculate_r2(y_test, test_pred)
    test_mse = calculate_mse(y_test, test_pred)
    
    return train_r2, test_r2, train_mse, test_mse

def preprocess_classification_data(df):
    df['temp_rolling_5'] = df['cpu_temp_C'].rolling(window=5).mean().fillna(df['cpu_temp_C'])
    df["activity"] = df["activity"].map({"idle": 0, "light_load":1, "medium_load":2 })
    df["surface_type"] = df["surface_type"].map({"soft":0, "rough":1})
    df["cpu_boost_mode"] = df["cpu_boost_mode"].map({"disabled":0, "aggressive":1})
    df["throttle"] = ((df["cpu_temp_C"] > 90) & (df["cpu_freq_MHz"] < 2000)).astype(int)
    features = ["activity",	"surface_type",	
            "cpu_boost_mode",
            "cpu_temp_slope",	"cpu_power_W",
            "cpu_util_pct",	
            "ram_used_GB",	"gpu_temp_C",
            "gpu_temp_slope",	"gpu_util_pct",
            "gpu_clock_MHz",	"gpu_power_W", 
            "cpu_temp_C", 'temp_rolling_5']

    X = df[features].values
    y = df["throttle"].values
    return X, y, features
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

def confusion_matrix(y_true, y_pred):
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








import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.utils import quick_fit 

def plot_confusion_matrix(y_true, y_pred):
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    
    for i, true_val in enumerate(unique_classes):
        for j, pred_val in enumerate(unique_classes):
            cm[i, j] = np.sum((y_true == true_val) & (y_pred == pred_val))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(unique_classes)
    ax.set_yticklabels(unique_classes)
    
    return unique_classes, cm, fig, ax

def test_temporal_degradation(model, df, features, prediction_window):
    print("\n" + "=" * 55)
    print("TEST 1: Temporal Degradation")
    print("=" * 55)
 
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
 
    
    fixed_split = int(len(df) * 0.8)
    test_df = df.iloc[fixed_split:]
    X_te = test_df[features].values
    y_te = test_df["target_future_throttle"].values
 
    results = []
    for frac in fractions:
        cutoff = int(len(df) * frac) - prediction_window
        if cutoff < 50:
            print(f"  frac={frac:.1f} → too few training rows, skipping")
            continue
        train_df = df.iloc[:cutoff]
        X_tr = train_df[features].values
        y_tr = train_df["target_future_throttle"].values
 
        _, acc, auc = quick_fit(model, X_tr, y_tr, X_te, y_te)
        results.append((frac, acc, auc))
        print(f"  Train frac={frac:.1f} | rows={cutoff:>5} | Acc={acc:.4f} | AUC={auc:.4f}")
 
    
    fracs, accs, aucs = zip(*results)
    plt.figure(figsize=(8, 4))
    plt.plot(fracs, accs, marker="o", label="Accuracy")
    plt.plot(fracs, aucs, marker="s", label="AUC")
    plt.xlabel("Training Fraction")
    plt.ylabel("Score")
    plt.title("Test 1 — Temporal Degradation\n(should rise with more training data)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test1_temporal_degradation.png", dpi=150)
    plt.show()
    print("  → Plot saved: test1_temporal_degradation.png")

     
def test_walk_forward(model, df, features, prediction_window):
    print("\n" + "=" * 55)
    print("TEST 4: Walk-Forward Validation")
    print("=" * 55)
 
    cutoffs = [0.5, 0.6, 0.7, 0.8]
    accs, aucs = [], []
 
    print(f"  {'Train cutoff':<14} {'Train rows':>11} {'Test rows':>10} {'Acc':>8} {'AUC':>8}")
    print(f"  {'-'*14} {'-'*11} {'-'*10} {'-'*8} {'-'*8}")
 
    for cutoff in cutoffs:
        split = int(len(df) * cutoff)
        window_size = 15000
        train_start = max(0, split - prediction_window - window_size)
        train_df = df.iloc[train_start:split - prediction_window]
        test_df  = df.iloc[split:]
 
        if len(test_df) < 50 or len(train_df) < 50:
            print(f"  cutoff={cutoff:.1f} → not enough rows, skipping")
            continue
 
        X_tr = train_df[features].values
        y_tr = train_df["target_future_throttle"].values
        X_te = test_df[features].values
        y_te = test_df["target_future_throttle"].values
 
        _, acc, auc = quick_fit(model, X_tr, y_tr, X_te, y_te)
        accs.append(acc)
        aucs.append(auc)
        print(f"  cutoff={cutoff:.1f}       {len(train_df):>11} {len(test_df):>10} {acc:>8.4f} {auc:>8.4f}")
 
 
    
    plt.figure(figsize=(8, 4))
    plt.plot(cutoffs[:len(accs)], accs, marker="o", label="Accuracy")
    plt.plot(cutoffs[:len(aucs)], aucs, marker="s", label="AUC")
    plt.xlabel("Train Cutoff Fraction")
    plt.ylabel("Score")
    plt.title("Test 4 — Walk-Forward Validation\n(should be consistent across folds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test4_walk_forward.png", dpi=150)
    plt.show()
    print("  → Plot saved: test4_walk_forward.png")
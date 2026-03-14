
import numpy as np
from collections import Counter
 
class DecisionTreeClassification:
    def __init__(self, max_depth=5, min_samples_split=50, min_samples_leaf=10, max_features='sqrt'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None
        

    def fit (self, X, y):
        self.tree = self._build_tree(X, y)

    def _gini(self, y):
        m = len(y)
        if m == 0: return 0
        counts = np.bincount(y)
        probabilities = counts / m
        return 1.0 - np.sum(probabilities**2)
    
    def _most_common_label(self, y):
        if len(y) == 0: return None
        return Counter(y).most_common(1)[0][0]

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)
        
        if (depth >= self.max_depth or 
            num_samples < self.min_samples_split or 
            len(unique_classes) == 1):
            return self._most_common_label(y)
        
        if self.max_features == 'sqrt':
            k = max(1, int(np.sqrt(num_features)))
        else:
            k = max(1, int(num_features * self.max_features))
        feat_idxs = np.random.choice(num_features, k, replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return self._most_common_label(y)
        
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] >best_thresh

        return{
            'feature': best_feat,
            'threshold': best_thresh,
            'left': self._build_tree(X[left_idx], y[left_idx], depth +1),
            'right': self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }
    
    def _best_split(self, X, y, feat_idxs):
        best_gini_gain = -1
        split_idx, split_thresh = None, None
        parent_gini = self._gini(y)

        for feat in feat_idxs:
            X_column = X[:, feat]
            thresholds = np.percentile(X_column, np.linspace(5, 95, 20))

            for thresh in thresholds:
                left_idx = X_column <= thresh
                right_idx = X_column > thresh

                # FIXED: Force the split search to respect min_samples_leaf
                if np.sum(left_idx) < self.min_samples_leaf or np.sum(right_idx) < self.min_samples_leaf:
                    continue

                n = len(y)
                n_l, n_r = np.sum(left_idx), np.sum(right_idx)
                gini_l, gini_r = self._gini(y[left_idx]), self._gini(y[right_idx])
                child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r

                gini_gain = parent_gini - child_gini

                if gini_gain > best_gini_gain:
                    best_gini_gain = gini_gain
                    split_idx = feat
                    split_thresh = thresh

        return split_idx, split_thresh
  
    
    def predict_batch(self, X, tree):
        if not isinstance(tree, dict):
            return np.full(X.shape[0], tree)
        
        left_mask = X[:, tree['feature']] <= tree['threshold']
        result = np.empty(X.shape[0])

        if left_mask.any():
            result[left_mask] = self.predict_batch(X[left_mask], tree['left'])
        if (~left_mask).any():
            result[~left_mask] = self.predict_batch(X[~left_mask], tree['right'])
    
        return result
            
    
class RandomForestClassifier:
    def __init__(self, n_trees=30, max_depth=4, min_samples_leaf=50, min_samples_split=100):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.trees = []
        

    def fit(self, X, y):
        self.trees = []
        y = y.astype(int)
        classes, counts = np.unique(y, return_counts=True)
        weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weights = np.array([weight_map[label] for label in y])
        sample_weights /= sample_weights.sum() 
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), len(X), replace=True, p=sample_weights)
            tree = DecisionTreeClassification(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X, threshold=0.75):
        all_preds = np.array([t.predict_batch(X, t.tree) for t in self.trees])

        probs = np.mean(all_preds, axis=0)
    
        return (probs >= threshold).astype(int)
    def predict_proba(self, X):
        all_preds = np.array([t.predict_batch(X, t.tree) for t in self.trees])

        # Average them to get the probability of class 1 (target_future_throttle)
        class_1_probs = np.mean(all_preds, axis=0)
        
        # The probability of class 0 is just the inverse
        class_0_probs = 1.0 - class_1_probs
        
        # Stack them side-by-side: Column 0 is class 0, Column 1 is class 1
        return np.column_stack((class_0_probs, class_1_probs))
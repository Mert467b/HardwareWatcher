
import numpy as np
from collections import Counter

class DecisionTreeRegression:
    def __init__(self, max_depth=15, min_samples_split=5, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_importance = None

    def fit (self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if depth >= self.max_depth or num_samples < self.min_samples_split or num_samples < self.min_samples_leaf:
            return np.mean(y)
        
        feat_idxs = np.random.choice(num_features, int(num_features * 0.5), replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return np.mean(y)
        
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] >best_thresh

        if np.sum(left_idx) < self.min_samples_leaf or np.sum(right_idx) < self.min_samples_leaf:
            return np.mean(y)
        return{
            'feature': best_feat,
            'threshold': best_thresh,
            'left': self._build_tree(X[left_idx], y[left_idx], depth +1),
            'right': self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }
    
    def _best_split(self, X, y, feat_idxs):
        best_mse = float('inf')
        split_idx, split_thresh = None, None
        n = len(y)

        for feat in feat_idxs:
            sort_indices = np.argsort(X[:, feat])
            X_sorted = X[sort_indices, feat]
            y_sorted = y[sort_indices]

            sum_l, sum_sq_l = 0.0, 0.0
            sum_r, sum_sq_r = np.sum(y_sorted), np.sum(y_sorted**2)
        
            for i in range(0, n - 1):
                val = y_sorted[i]
                sum_l += val
                sum_sq_l += val**2
                sum_r -= val
                sum_sq_r -= val**2

                if X_sorted[i] < X_sorted[i+1]:
                    n_l = i + 1
                    n_r = n - n_l
                
                    mse_l = sum_sq_l - (sum_l**2 / n_l)
                    mse_r = sum_sq_r - (sum_r**2 / n_r)
                    current_mse = mse_l + mse_r

                    if current_mse < best_mse:
                        best_mse = current_mse
                        split_idx = feat
                        split_thresh = (X_sorted[i] + X_sorted[i+1]) / 2.0

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
            
    
class RandomForestRegressor:
    def __init__(self, n_trees=15, max_depth=15, min_samples_leaf=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTreeRegression(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
            self._accumulate_importance(tree.tree)
        
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()

    def _accumulate_importance(self, node):
        if isinstance(node, dict):
            self.feature_importances_[node['feature']] += 1
            self._accumulate_importance(node['left'])
            self._accumulate_importance(node['right'])

    def predict(self, X):
        all_preds = np.array([t.predict_batch(X, t.tree) for t in self.trees])
        return all_preds.mean(axis=0)
    
class DecisionTreeClassification:
    def __init__(self, max_depth=15, min_samples_split=5, min_samples_leaf=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
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
        
        feat_idxs = np.random.choice(num_features, int(num_features * 0.5), replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return self._most_common_label(y)
        
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] >best_thresh

        if np.sum(left_idx) < self.min_samples_leaf or np.sum(right_idx) < self.min_samples_leaf:
            return self._most_common_label(y)
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

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
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
    def __init__(self, n_trees=30, max_depth=15, min_samples_leaf=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        

    def fit(self, X, y):
        y = y.astype(int)
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), len(X), replace=True)
            tree = DecisionTreeClassification(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X, threshold=0.3):
        all_preds = np.array([t.predict_batch(X, t.tree) for t in self.trees])

        probs = np.mean(all_preds, axis=0)
    
        return (probs >= threshold).astype(int)
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

class RandomForest(BaseEstimator):
    def __init__(self,
                 tree_type,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 random_state=None,
                 warm_start=False,
                 class_weight=None):

        self.tree_type = tree_type
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.class_weight = class_weight

        self.trees = []

    def fit(self, X, y):
        for i in range(self.n_estimators):
            if self.tree_type == 'decision':
                tree = \
                    DecisionTreeClassifier(criterion = self.criterion,
                        max_depth = self.max_depth,
                        min_samples_split = self.min_samples_split,
                        min_samples_leaf = self.min_samples_leaf,
                        min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                        max_features = self.max_features,
                        random_state = self.random_state,
                        max_leaf_nodes = self.max_leaf_nodes,
                        min_impurity_decrease = self.min_impurity_decrease,
                        min_impurity_split = self.min_impurity_split,
                        class_weight = self.class_weight)
            elif self.tree_type == 'extra':
                tree = \
                    ExtraTreeClassifier(criterion = self.criterion,
                        max_depth = self.max_depth,
                        min_samples_split = self.min_samples_split,
                        min_samples_leaf = self.min_samples_leaf,
                        min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                        max_features = self.max_features,
                        random_state = self.random_state,
                        max_leaf_nodes = self.max_leaf_nodes,
                        min_impurity_decrease = self.min_impurity_decrease,
                        min_impurity_split = self.min_impurity_split,
                        class_weight = self.class_weight)

            Xs, ys = resample(X, y, replace=self.bootstrap,
                        random_state = self.random_state)

            tree.fit(Xs, ys);

            self.trees.append(tree)

        return self

    def predict(self, X):
        y = np.zeros((X.shape[0],))

        for tree in self.trees:
            y += tree.predict(X);

        y /= self.n_estimators;
        y[y >= 0.5] = 1;
        y[y <  0.5] = 0;

        return y.astype(int);

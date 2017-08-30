from __future__ import division, print_function

import numpy as np


def _sample_axes(labels, n_samples=1):
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    for _ in range(n_samples):
        yield np.random.choice(pos), np.random.choice(neg)


def _split_metric(total_left, total_right, true_left, true_right):
    left_pred = true_left / total_left
    right_pred = true_right / total_right

    left_gini = 1 - left_pred**2 - (1 - left_pred)**2
    right_gini = 1 - right_pred ** 2 - (1 - right_pred)**2

    left_prop = total_left / (total_left + total_right)
    return left_prop * left_gini + (1 - left_prop) * right_gini


class Node:
    def __init__(self, depth, similarity_function=np.dot, n_axes=1,
                 max_depth=None):
        self.depth = depth
        self.max_depth = max_depth
        self._sim = similarity_function
        self.n_axes = n_axes
        self._left = None
        self._right = None
        self._p = None
        self._q = None
        self.criterion = None
        self.prediction = None

    def _find_split(self, X, y, p, q):
        sims = [self._sim(x, q) - self._sim(x, p) for x in X]
        indices = sorted([i for i in range(len(y)) if not np.isnan(sims[i])],
                         key=lambda x: sims[x])

        best_metric = 1
        best_p = None
        best_q = None
        best_criterion = 0

        n = len(indices)
        total_true = sum([y[j] for j in indices])
        left_true = 0
        for i in range(n - 1):
            left_true += y[indices[i]]
            right_true = total_true - left_true
            split_metric = _split_metric(i + 1, n - i - 1, left_true, right_true)
            if split_metric < best_metric:
                best_metric = split_metric
                best_p = p
                best_q = q
                best_criterion = (sims[indices[i]] + sims[indices[i + 1]]) / 2
        return best_metric, best_p, best_q, best_criterion

    def fit(self, X, y):
        self.prediction = sum(y) / len(y)
        if self.prediction in [0, 1]:
            return self

        if self.max_depth is not None and self.depth >= self.max_depth:
            return self

        best_metric = 1
        best_p = None
        best_q = None
        best_criterion = 0
        for i, j in _sample_axes(y, self.n_axes):
            metric, p, q, criterion = self._find_split(X, y, X[i], X[j])
            if metric < best_metric:
                best_metric = metric
                best_p = p
                best_q = q
                best_criterion = criterion

        # Split found
        if best_metric < 1:
            self._p = best_p
            self._q = best_q
            self.criterion = best_criterion

            sims = [self._sim(x, self._q) - self._sim(x, self._p) for x in X]
            X_left = X[sims <= self.criterion, :]
            X_right = X[sims > self.criterion, :]
            y_left = y[sims <= self.criterion]
            y_right = y[sims > self.criterion]

            if len(y_left) > 0 and len(y_right) > 0:
                self._left = Node(
                    self.depth + 1, self._sim, self.n_axes, self.max_depth).fit(X_left, y_left)
                self._right = Node(
                    self.depth + 1, self._sim, self.n_axes, self.max_depth).fit(X_right, y_right)

        return self

    def _predict_proba_once(self, x):
        if self._left is None:
            return self.prediction
        elif self._sim(x, self._q) - self._sim(x, self._p) <= self.criterion:
            return self._left._predict_proba_once(x)
        elif self._sim(x, self._q) - self._sim(x, self._p) > self.criterion:
            return self._right._predict_proba_once(x)
        else:
            return self.prediction

    def predict_proba(self, X):
        return [self._predict_proba_once(x) for x in X]


class SimilarityForest:
    """
    Basic implementation of SimForest, as outlined in
    'Similarity Forests', S. Sathe and C. C. Aggarwal, KDD 2017'.

    :param n_estimators: number of trees in the forest (default=10)
    :param similarity_function: similarity function (default is dot product) -
                                should return np.nan if similarity unknown
    :param n_axes: number of 'axes' per split
    :param max_depth: maximum depth to grow trees to (default=None)
    """
    def __init__(self, n_estimators=10, similarity_function=np.dot, n_axes=1,
                 max_depth=None):
        self.n_estimators = n_estimators
        self.n_axes = n_axes
        self.max_depth = max_depth
        self._sim = similarity_function
        self._trees = None

    def _bag(self, X, y):
        selection = np.array(list(set(np.random.choice(len(y), size=len(y)))))
        return X[selection, :], y[selection]

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        :param X: training set
        :param y: training set labels
        :return: self
        """
        if len(X) != len(y):  # @@@ More checks
            print('Bad sizes: {}, {}'.format(X.shape, y.shape))
        else:
            self._trees = [
                Node(1, self._sim, self.n_axes, self.max_depth).fit(
                    *self._bag(X, y))
                for _ in range(self.n_estimators)
            ]
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities of X.

        :param X: samples to make prediction probabilities for
        :return: array of prediction probabilities for each class
        """
        probs = np.mean([t.predict_proba(X) for t in self._trees], axis=0)
        return np.c_[1 - probs, probs]

    def predict(self, X):
        """
        Predict class of X.

        :param X: samples to make predictions for
        :return: array of class predictions
        """
        return (self.predict_proba(X)[:, 1] > 0.5).astype(np.int)

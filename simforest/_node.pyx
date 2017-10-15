import numpy as np
cimport numpy as np


def _sample_axes(labels, rand, n_samples=1):
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    for _ in range(n_samples):
        yield rand.choice(pos), rand.choice(neg)


cdef double _split_metric(double total_left,
                          double total_right,
                          double true_left,
                          double true_right):
    cdef:
        double left_pred, right_pred, left_gini, right_gini, left_prop

    left_pred = true_left / total_left
    right_pred = true_right / total_right

    left_gini = 1 - left_pred**2 - (1 - left_pred)**2
    right_gini = 1 - right_pred ** 2 - (1 - right_pred)**2

    left_prop = total_left / (total_left + total_right)
    return left_prop * left_gini + (1 - left_prop) * right_gini


class Node:
    def __init__(self, depth, similarity_function=np.dot, n_axes=1,
                 max_depth=None, rand=None):
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
        self._rand = np.random.RandomState() if rand is None else rand

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
        for i, j in _sample_axes(y, self._rand, self.n_axes):
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
                self._left = Node(self.depth + 1,
                                  self._sim,
                                  self.n_axes,
                                  self.max_depth,
                                  self._rand).fit(X_left, y_left)
                self._right = Node(self.depth + 1,
                                   self._sim,
                                   self.n_axes,
                                   self.max_depth,
                                   self._rand).fit(X_right, y_right)

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
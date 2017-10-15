from __future__ import division, print_function

import numpy as np
from ._node import Node


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
                 max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.n_axes = n_axes
        self.max_depth = max_depth
        self._sim = similarity_function
        self._trees = None
        self._rand = np.random.RandomState(random_state)

    def _bag(self, X, y):
        selection = np.array(list(set(self._rand.choice(len(y), size=len(y)))))
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
            self._trees = [Node(1,
                                self._sim,
                                self.n_axes,
                                self.max_depth,
                                self._rand).fit(*self._bag(X, y))
                           for _ in range(self.n_estimators)]
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

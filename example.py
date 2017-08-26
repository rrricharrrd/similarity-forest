from simforest import SimilarityForest

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    X, y = make_blobs(n_samples=100, centers=[(0, 0), (1, 1)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

    clf = SimilarityForest(n_estimators=20, n_axes=2)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    prob = clf.predict_proba(X_test)

    print(prob[:, 1])
    print(y_test)
    print(accuracy_score(y_test, pred))

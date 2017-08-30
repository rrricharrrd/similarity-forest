from simforest import SimilarityForest

from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    X, y = make_blobs(n_samples=1000, centers=[(0, 0), (1, 1)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

    sf = SimilarityForest(n_estimators=20, n_axes=1)
    sf.fit(X_train, y_train)

    sf_pred = sf.predict(X_test)
    sf_prob = sf.predict_proba(X_test)

    print('Similarity Forest')
    print(sf_prob[:, 1])
    print(y_test)
    print(accuracy_score(y_test, sf_pred))

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)

    print('Random Forest')
    print(rf_prob[:, 1])
    print(y_test)
    print(accuracy_score(y_test, rf_pred))

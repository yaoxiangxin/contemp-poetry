import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PARAM_RANGE = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
FILE = 'extended.csv'
df = pd.read_csv(FILE)
feature_names = list(df.columns[1:]) # for export_grashviz and feature_importances_
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


# custom-made cross_val_score
def my_cross_val_score(estimator, X, y, cv=5,
                       gs=True, tree=False, forest=False):
    train_scores, test_scores = [], []
    for i, (train, test) in enumerate(StratifiedKFold(y=y, n_folds=cv)):
        estimator.fit(X=X[train], y=y[train])

        if gs: # isinstance(estimator, GridSearchCV)
            print(estimator.best_params_)

        if tree: # isinstance(estimator, DecisionTreeClassifier)
            export_graphviz(estimator.best_estimator_,
                            out_file='tree-{}-{}.dot'.format(FILE[:-4], i),
                            feature_names=feature_names)

        if forest: # isinstance(estimator, RandomForestClassifier)
            feature_importances = estimator.best_estimator_.feature_importances_
            for i in reversed(np.argsort(feature_importances)):
                print('{:19} : {:.2%}'.format(feature_names[i],
                                              feature_importances[i]))

        train_scores.append(estimator.score(X=X[train], y=y[train]))
        test_scores.append(estimator.score(X=X[test], y=y[test]))
    return train_scores, test_scores


def my_print(train_scores, test_scores):
    print('CV training accuracy : {:.2%} +/- {:.2%}'.format(np.mean(train_scores),
                                                            np.std(train_scores)))
    print('CV     test accuracy : {:.2%} +/- {:.2%}'.format(np.mean(test_scores),
                                                            np.std(test_scores)))

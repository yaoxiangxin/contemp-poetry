from sklearn.decomposition import PCA
from contemppoetry import *


print('******PCA-LogisticRegression')
pipe = Pipeline([('scl', StandardScaler()),
                 ('pca', PCA()),
                 ('clf', LogisticRegression())])
param_grid = {'pca__n_components': [8, 9, 10],
              'clf__penalty': ['l1', 'l2'],
              'clf__C': PARAM_RANGE}
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))


print('******PCA-SVC')
pipe = Pipeline([('scl', StandardScaler()),
                 ('pca', PCA()),
                 ('clf', SVC())])
param_grid = [{'pca__n_components': [8, 9, 10],
               'clf__kernel': ['linear'],
               'clf__C': PARAM_RANGE},
              {'pca__n_components': [8, 9, 10],
			   'clf__kernel': ['rbf'],
			   'clf__C': PARAM_RANGE,
			   'clf__gamma': PARAM_RANGE}]
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))

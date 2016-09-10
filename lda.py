from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from contemppoetry import *


print('******LinearDiscriminantAnalysis-LogisticRegression')
pipe = Pipeline([('scl', StandardScaler()),
                 ('lda', LinearDiscriminantAnalysis()),
                 ('clf', LogisticRegression())])
param_grid = {'lda__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'clf__penalty': ['l1', 'l2'],
              'clf__C': PARAM_RANGE}
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))


print('******LinearDiscriminantAnalysis-SVC')
pipe = Pipeline([('scl', StandardScaler()),
                 ('lda', LinearDiscriminantAnalysis()),
                 ('clf', SVC())])
param_grid = [{'lda__n_components': [1, 2, 3, 4, 5, 6],
               'clf__kernel': ['linear'],
               'clf__C': PARAM_RANGE},
			  {'lda__n_components': [1, 2, 3, 4, 5, 6],
			    'clf__kernel': ['rbf'],
			    'clf__C': PARAM_RANGE,
			    'clf__gamma': PARAM_RANGE}]
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))


print('******LinearDiscriminantAnalysis-DecisionTreeClassifier')
pipe = Pipeline([('scl', StandardScaler()),
                 ('lda', LinearDiscriminantAnalysis()),
                 ('clf', DecisionTreeClassifier())])
param_grid = {'lda__n_components': [1, 2, 3, 4, 5, 6],
              'clf__criterion': ['gini', 'entropy'],
              'clf__max_depth': [1, 2, 4, 8, 16, None]}
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))


print('******LinearDiscriminantAnalysis-KNeighborsClassifier')
pipe = Pipeline([('scl', StandardScaler()),
                 ('lda', LinearDiscriminantAnalysis()),
                 ('clf', KNeighborsClassifier())])
param_grid = {'lda__n_components': [1, 2, 3, 4, 5, 6],
              'clf__n_neighbors': [5, 11, 23],
              'clf__p': [1, 2]}
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))

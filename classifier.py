from contemppoetry import *


print('******LogisticRegression')
pipe = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression())])
param_grid = {'clf__penalty': ['l1', 'l2'],
              'clf__C': PARAM_RANGE}
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))


print('******SVC')
pipe = Pipeline([('scl', StandardScaler()), ('clf', SVC())])
param_grid = [{'clf__kernel': ['linear'],
               'clf__C': PARAM_RANGE},
			  {'clf__kernel': ['rbf'],
			   'clf__C': PARAM_RANGE,
               'clf__gamma': PARAM_RANGE}]
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))


print('******DecisionTreeClassifier')
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [1, 2, 4, 8, 16, None]}
gs = GridSearchCV(estimator=DecisionTreeClassifier(),
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y, tree=True))


print('******KNeighborsClassifier')
pipe = Pipeline([('scl', StandardScaler()), ('clf', KNeighborsClassifier())])
param_grid = {'clf__n_neighbors': [5, 11, 23],
              'clf__p': [1, 2]}
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from contemppoetry import *


print('******RandomForestClassifier')
param_grid = {'n_estimators': [10, 20, 40, 80, 160, 320],
			  'max_depth': [1, 2, 4, 8, 16, None]}
gs = GridSearchCV(estimator=RandomForestClassifier(),
				  param_grid=param_grid, scoring='accuracy', cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y, forest=True))


print('******AdaBoostClassifier')
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree)
param_grid = {'n_estimators': [500, 1000, 2000, 5000],
              'learning_rate': [0.001, 0.01, 0.1, 1.0, 10.0]}
gs = GridSearchCV(estimator=ada,
				  param_grid=param_grid,
				  scoring='accuracy',
				  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))


print('******VotingClassifier')
vclf = VotingClassifier(estimators=[('lr', LogisticRegression()),
                                    ('svc', SVC()),
                                    ('knn', KNeighborsClassifier())])
pipe = Pipeline([('scl', StandardScaler()), ('clf', vclf)])
param_grid = {'clf__lr__C': PARAM_RANGE,
              'clf__svc__C': PARAM_RANGE,
			  'clf__svc__gamma': PARAM_RANGE,
			  'clf__knn__n_neighbors': [5, 11]}
gs = GridSearchCV(estimator=pipe,
				  param_grid=param_grid,
				  scoring='accuracy',
				  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))

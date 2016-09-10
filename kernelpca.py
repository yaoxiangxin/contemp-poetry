from sklearn.decomposition import KernelPCA
from contemppoetry import *


print('******KernelPCA-LogisticRegression')
pipe = Pipeline([('scl', StandardScaler()),
                 ('kpca', KernelPCA()),
                 ('clf', LogisticRegression())])
param_grid = {'kpca__n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'kpca__gamma': PARAM_RANGE,
              'clf__penalty': ['l1', 'l2'],
              'clf__C': PARAM_RANGE}
gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5)
my_print(*my_cross_val_score(gs, X=X, y=y))

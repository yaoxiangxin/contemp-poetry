import itertools
from sklearn.feature_selection import RFECV
from contemppoetry import *


print('******RFECV-LogisticRegression')
for penalty, C in itertools.product(['l1', 'l2'], PARAM_RANGE):
    rfe = RFECV(estimator=LogisticRegression(penalty=penalty, C=C),
                scoring='accuracy', cv=5)
    rfe.fit(X, y)

    # list selected features by rank
    print(
        [feature_names[i] for i in np.argsort(rfe.ranking_) if rfe.support_[i]]
    )

    pipe = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression())])
    param_grid = {'clf__penalty': ['l1', 'l2'],
                  'clf__C': PARAM_RANGE}
    gs = GridSearchCV(estimator=pipe,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=5)
    my_print(*my_cross_val_score(gs, X=rfe.transform(X), y=y, gs=False))

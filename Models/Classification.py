from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def logistic_regression(X_train, y_train, C=1.0, max_iter=100):
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def random_forest_classifier(X_train, y_train, n_estimators=100, max_depth=None):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def xgboost_classifier(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
    model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def svc_classifier(X_train, y_train, kernel='rbf', C=1.0):
    model = SVC(kernel=kernel, C=C, probability=True)
    model.fit(X_train, y_train)
    return model

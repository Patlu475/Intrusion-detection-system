"""Random Forest classifier for intrusion detection."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2'],
}

DEFAULT_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'max_features': 'sqrt',
}


def build_model(params=None):
    p = params or DEFAULT_PARAMS
    return RandomForestClassifier(random_state=42, n_jobs=-1, **p)


def train_with_gridsearch(X_train, y_train, cv=3):
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    gs = GridSearchCV(
        base, PARAM_GRID, cv=cv, scoring='f1_macro',
        n_jobs=-1, verbose=1, refit=True,
    )
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV macro F1: {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_


def train(X_train, y_train, params=None):
    model = build_model(params)
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return model.predict(X)


def predict_proba(model, X):
    return model.predict_proba(X)

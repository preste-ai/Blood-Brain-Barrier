import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate


class ForestClassifier:

    def __init__(self, class_weight, n_estimators, min_samples_leaf):
        self.class_weight = class_weight
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.model = None

    def make_classifier(self, X, y):
        self.model = RandomForestClassifier(class_weight=self.class_weight,
                                            n_estimators=self.n_estimators,
                                            min_samples_leaf=self.min_samples_leaf,
                                            random_state=42)
        self.model.fit(X, y)
        return self.model

    def cross_validate(self, X, y, cv):

        self.model = self.make_classifier(X, y)

        cv = cross_validate(estimator=self.model,
                            X=X,
                            y=y,
                            cv=cv,
                            scoring=['precision', 'recall'],
                            return_train_score=True)

        return cv['train_precision'].mean(), \
               cv['train_recall'].mean(), \
               cv['test_precision'].mean(), \
               cv['test_recall'].mean()

    def validate(self, X_val, y_val):
        return classification_report(y_val, self.model.predict(X_val))
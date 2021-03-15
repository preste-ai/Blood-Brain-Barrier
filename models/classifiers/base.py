import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold


class BaseClassifier:

    def __init__(self, name):

        """
        Base classifier class
        :param name: name of the classifier (str)
        """

        self.name = name

    def fit(self, X, y):

        """
        Fit model
        :param X: array with molecule features and their values
        :param y: array with target values
        :return: model fitted into data
        """

        self.model.fit(X, y)

        return self.model

    def cross_validate(self, X, y, n_splits):

        """
        Cross-validate model
        :param X: array with molecule features and their values
        :param y: array with target values
        :param n_splits: number of folds to split data into for cross-validation
        :return: mean values of train and test scores
        """

        skf = StratifiedKFold(n_splits=n_splits)

        reports_train = []
        reports_test = []

        for train, test in skf.split(X, y):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]

            self.model = self.fit(X_train, y_train)

            target_names = ['class 0', 'class 1']
            report_train = pd.DataFrame(classification_report(y_true=y_train,
                                                              y_pred=self.model.predict(X_train),
                                                              output_dict=True,
                                                              target_names=target_names))
            report_train = report_train.loc[['precision', 'recall'], ['class 0', 'class 1']]

            report_test = pd.DataFrame(classification_report(y_true=y_test,
                                                             y_pred=self.model.predict(X_test),
                                                             output_dict=True,
                                                             target_names=target_names))
            report_test = report_test.loc[['precision', 'recall'], ['class 0', 'class 1']]

            reports_train.append(report_train)
            reports_test.append(report_test)

        reports_train = pd.concat(reports_train)
        reports_test = pd.concat(reports_test)

        return reports_train.groupby(reports_train.index).mean(), reports_test.groupby(reports_test.index).mean()

    def validate(self, X_val, y_val):

        """
        Validate model
        :param X_val: validation data set with molecule features and their values
        :param y_val: validation data set with target values
        :return: classification report for model predictions on validation set
        """

        report = pd.DataFrame(classification_report(y_true=y_val,
                                                    y_pred=self.model.predict(X_val),
                                                    output_dict=True)).T

        return report

    def predict(self, X):
        """
        Validate model
        :param X: data set with molecule features and their values
        :return:
        """

        predictions = self.model.predict(X)

        return predictions
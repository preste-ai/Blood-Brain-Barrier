import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold


class BaseClassifier:

    def __init__(self, name):
        self.name = name

    def fit(self, X, y):

        """

        :param X:
        :param y:
        :return:
        """

        self.model.fit(X, y)

        return self.model

    def cross_validate(self, X, y, n_splits):

        """

        :param X:
        :param y:
        :param n_splits:
        :return:
        """

        skf = StratifiedKFold(n_splits=n_splits, random_state=self.random_state)

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

        :param X_val:
        :param y_val:
        :return:
        """

        report = pd.DataFrame(classification_report(y_true=y_val,
                                                    y_pred=self.model.predict(X_val),
                                                    output_dict=True)).T

        return report
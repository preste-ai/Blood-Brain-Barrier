import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


class Transformer:

    def __init__(self, X, y, n):

        """

        :param X:
        :param y:
        :param n:
        """

        self.X = X
        self.y = y
        self.n = n

    def transform(self):

        """
        Initialize and fit PCA model with n components
        :return: transformed data
        """

        model = PCA(n_components=self.n)
        transformed = model.fit_transform(self.X, self.y)

        return transformed

    def find_best_n(self, number, class_weights):

        """
        define best number of components
        :param number: max number of components to consider
        :return:
        """

        precisions = []
        recalls = []
        n_components = list(range(1, number))

        for n in n_components:

            transformer = PCA(n_components=n)
            transformed = pd.DataFrame(transformer.fit_transform(self.X))

            classifier = LogClassifier(class_weights=class_weights)

            precision, recall = classifier.\
                optimize_precision_recall(X=transformed, y=self.y, n_splits=5)

            precisions.append(precision)
            recalls.append(recall)

        plt.plot(n_components, precisions, '-o', label='Precision')
        plt.plot(n_components, recalls, '-o', label='Recall')
        plt.legend()
        plt.title('Precision & Recall for different number of PCs used for LR model')
        plt.show()

    def plot_transformed(self):

        """

        :return:
        """

        model = PCA(n_components=2)
        transformed = model.fit_transform(self.X, self.y)

        pca_df = pd.DataFrame({'y': self.y,
                               'PC1': transformed[:, 0],
                               'PC2': transformed[:, 1]})

        plt.figure()
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="y", s=70)
        plt.title('PCA transformed data')
        plt.show()


class LogClassifier:

    def __init__(self, class_weights):

        """

        :param class_weights:
        """

        self.class_weights = class_weights
        self.model = None

    def make_classifier(self, X, y):

        """

        :param X:
        :param y:
        :return:
        """

        self.model = LogisticRegression(class_weight=self.class_weights)
        self.model.fit(X, y)
        return self.model

    def cross_validate(self, X, y, n_splits):

        """

        :param X:
        :param y:
        :param cv:
        :return:
        """

        skf = StratifiedKFold(n_splits=n_splits)

        reports_train = []
        reports_test = []

        for train, test in skf.split(X, y):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]

            self.model = self.make_classifier(X_train, y_train)

            target_names = ['class 0', 'class 1']
            report_train = pd.DataFrame(classification_report(y_train, self.model.predict(X_train),
                                                              output_dict=True,
                                                              target_names=target_names))
            report_train = report_train.loc[['precision', 'recall'], ['class 0', 'class 1']]

            report_test = pd.DataFrame(classification_report(y_test, self.model.predict(X_test),
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

        return classification_report(y_val, self.model.predict(X_val))

    def optimize_precision_recall(self, X, y, n_splits):

        self.model = self.make_classifier(X, y)

        cv = cross_validate(estimator=self.model,
                            X=X,
                            y=y,
                            cv=n_splits,
                            scoring=['precision', 'recall'],
                            return_train_score=True)

        return cv['test_precision'].mean(), \
               cv['test_recall'].mean()

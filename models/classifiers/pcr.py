import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from models.classifiers.base import BaseClassifier


class Transformer:

    def __init__(self, n):

        """
        PCA data transformer class
        :param n: number of principal components to construct
        """

        self.n = n
        self.model = PCA(n_components=self.n)

    def transform(self, X, y):

        """
        Initialize and fit PCA model with n components
        :param X: array with molecule features and their values
        :param y: array with target values
        :return: transformed data
        """

        transformed = self.model.fit_transform(X, y)

        return transformed

    def find_best_n(self, X, y, number, class_weights):

        """
        Define best number of components using Logistic Regression precision and recall scores
        :param X: array with molecule features and their values
        :param y: array with target values
        :param number: max number of components to consider
        :param class_weights: dictionary with weights for different classes for Logistic Regression
        :return: plot with precision and recall scores for Logistic Regression
                 fitted into data with different number of components
        """

        precisions = []
        recalls = []
        n_components = list(range(1, number))

        for n in n_components:

            transformer = PCA(n_components=n)
            transformed = pd.DataFrame(transformer.fit_transform(X))

            classifier = LogClassifier(class_weights=class_weights)

            precision, recall = classifier.\
                get_precision_recall(X=transformed, y=y, n_splits=5)

            precisions.append(precision)
            recalls.append(recall)

        plt.plot(n_components, precisions, '-o', label='Precision')
        plt.plot(n_components, recalls, '-o', label='Recall')
        plt.legend()
        plt.title('Precision & Recall for different number of PCs used for LR model')
        plt.show()

    def plot_transformed(self, X, y):

        """
        Plot data set PCA-transformed data
        :param X: array with molecule features and their values
        :param y: array with target values
        :return: plot with data transformed into 2D with PCA, data points color-coded by classes
        """

        model = PCA(n_components=2)
        transformed = model.fit_transform(X, y)

        pca_df = pd.DataFrame({'y': y,
                               'PC1': transformed[:, 0],
                               'PC2': transformed[:, 1]})

        plt.figure()
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="y", s=70)
        plt.title('PCA transformed data')
        plt.show()


class LogClassifier(BaseClassifier):

    def __init__(self, class_weights):

        """
        Logistic Regression model class
        :param class_weights: dictionary with weights for different classes
        """

        super(BaseClassifier, self).__init__()
        self.class_weights = class_weights
        self.model = LogisticRegression(class_weight=self.class_weights)

    def get_precision_recall(self, X, y, n_splits):

        """
        Get precision and recall scores with cross-validation
        :param X: array with molecule features and their values
        :param y: array with target values
        :param n_splits: number of folds to split data into for cross-validation
        :return: mean values of cross-validated precision and recall scores
        """

        cv = cross_validate(estimator=self.model,
                            X=X,
                            y=y,
                            cv=n_splits,
                            scoring=['precision', 'recall'],
                            return_train_score=True)

        return cv['test_precision'].mean(), cv['test_recall'].mean()

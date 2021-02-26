import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from models.classifiers.base import BaseClassifier


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

            classifier = LogClassifier(class_weights=class_weights,
                                       random_state=42)

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


class LogClassifier(BaseClassifier):

    def __init__(self, class_weights, random_state):

        """

        :param class_weights:
        """

        super(BaseClassifier, self).__init__()
        self.class_weights = class_weights
        self.random_state = random_state
        self.model = LogisticRegression(class_weight=self.class_weights)

    def optimize_precision_recall(self, X, y, n_splits):

        """

        :param X:
        :param y:
        :param n_splits:
        :return:
        """

        cv = cross_validate(estimator=self.model,
                            X=X,
                            y=y,
                            cv=n_splits,
                            scoring=['precision', 'recall'],
                            return_train_score=True)

        return cv['test_precision'].mean(), cv['test_recall'].mean()

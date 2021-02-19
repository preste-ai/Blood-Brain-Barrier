import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


class Transformer:

    def __init__(self, X, y, n):
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

    def find_best_n(self, number):

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
            transformed = transformer.fit_transform(self.X)

            classifier = LogClassifier(class_weights={0: 1, 1: 5})

            _, precision, _, recall = classifier.cross_validate(X=transformed,
                                                                y=self.y,
                                                                cv=5)

            precisions.append(precision)
            recalls.append(recall)

        plt.plot(n_components, precisions, '-o', label='Precision')
        plt.plot(n_components, recalls, '-o', label='Recall')
        plt.legend()
        plt.title('Precision & Recall for different number of PCs used for LR model')
        plt.show()

    def plot_transformed(self):

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
        self.class_weights = class_weights
        self.model = None

    def make_classifier(self, X, y):
        self.model = LogisticRegression(class_weight=self.class_weights)
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
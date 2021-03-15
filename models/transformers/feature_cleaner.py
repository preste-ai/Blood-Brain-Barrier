import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class Selector:

    def __init__(self, target, n):

        """
        :param target: string indicating target name
        :param n: number of features to select with each algorithm in class
        """

        self.target = target
        self.n = n

    def clean(self, data, threshold):

        """
        Cleans NaN, inter correlated and zero variance features
        :param data: data set to clean
        :param threshold: maximum value allowed for Pearson correlation between features
        :return: clean data set split into X and y
        """

        X = data.drop(self.target, axis=1)
        y = data.loc[:, self.target].fillna(value=0)

        X = X.\
            infer_objects().\
            select_dtypes(include=['float64', 'int64']).\
            dropna(axis=1)

        # drop away nulls and zero variance
        for column in X:
            zero_var = X[column].std() == 0
            null = X[column].isna().sum() == X.shape[0]
            if zero_var or null:
                X.drop(columns=column, inplace=True)

        # drop r > 0.95 inter correlated features
        corr = np.sum(abs(X.corr()) > threshold, axis=1)
        corr_mask = corr <= 1
        X = X.loc[:, corr_mask]

        return X, y

    @staticmethod
    def correlating_features(data, threshold):

        """
        Detect and drop intercorrelated features
        :param data: data set to clean
        :param threshold: maximum value allowed for Pearson correlation between features
        :return: clean data
        """

        col_corr = set()  # Set of all the names of deleted columns
        corr_matrix = data.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
                    if colname in data.columns:
                        data = data.drop(colname, axis=1)  # deleting the column from the dataset
        return data

    def scale(self, data, standard=True):

        """
        Normalize features in data set
        :param data: data set to normalize
        :param standard: method used for normalization:
                         True for StandardScaler
                         False for MinMaxScaler
        :return: normalized data split into X and y
        """

        X, y = self.clean(data=data,
                          threshold=0.95)

        if standard:
            scaler = StandardScaler()
            scaled_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        else:
            scaler = MinMaxScaler()
            scaled_X = pd.DataFrame(scaler.fit_transform(X, y), columns=X.columns)

        return scaled_X, y

    def chi_selector(self, X, y):

        """
        CHI2 filter
        :param X: array with molecule features and their values
        :param y: array with target values
        :return: boolean series with True for selected features
        """

        chi_selector = SelectKBest(chi2, k=self.n)
        chi_selector.fit(X, y)
        chi_support = chi_selector.get_support()

        return chi_support

    def rfe_selector(self, X, y):

        """
        RFE with RF filter
        :param X: array with molecule features and their values
        :param y: array with target values
        :return: boolean series with True for selected features
        """

        rfe_selector = RFE(estimator=RandomForestClassifier(n_estimators=100,
                                                            min_samples_leaf=5,
                                                            random_state=42),
                           n_features_to_select=self.n,
                           step=20)
        rfe_selector.fit(X, y)
        rfe_support = rfe_selector.get_support()

        return rfe_support

    def lasso_selector(self, X, y):

        """
        LASSO filter
        :param X: array with molecule features and their values
        :param y: array with target values
        :return: boolean series with True for selected features
        """

        lr_selector = SelectFromModel(LogisticRegression(penalty="l1",
                                                         class_weight='balanced',
                                                         solver='liblinear',
                                                         multi_class='ovr'),
                                      max_features=self.n)
        lr_selector.fit(X, y)

        lr_support = lr_selector.get_support()

        return lr_support

    def select(self, data):

        """
        Combine all feature selection methods
        :param data: data to clean
        :return: normalized data split into X and y,
                 dataframe with votes of 3 selectors for each feature in the data set
        """

        X, y = self.scale(data=data, standard=False)
        chi_support = self.chi_selector(X, y)

        X, y = self.scale(data=data, standard=True)
        rfe_support = self.rfe_selector(X, y)
        lr_support = self.lasso_selector(X, y)

        features = X.columns

        selected = pd.DataFrame({'Feature': features,
                                 'Chi-2': chi_support,
                                 'RFE': rfe_support,
                                 'LASSO': lr_support})

        # count the selected times for each feature
        selected['Total'] = np.sum(selected, axis=1)
        selected = selected.sort_values(['Total', 'Feature'], ascending=False)

        return X, y, selected


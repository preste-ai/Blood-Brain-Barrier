from imblearn.over_sampling import SMOTE


class Sampler:

    def __init__(self, sampling_strategy, random_state):

        """
        :param sampling_strategy: fraction of minority class in the transformed data set
        """

        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def transform(self, X, y):

        """
        Add synthetic samples to the data set using SMOTE algorithm
        :param X: data frame with sample features of the data set
        :param y: series with target values of the data set
        :return: X and y with synthetic samples
        """

        model = SMOTE(sampling_strategy=self.sampling_strategy,
                      random_state=self.random_state)
        X_smo, y_smo = model.fit_resample(X, y)

        return X_smo, y_smo

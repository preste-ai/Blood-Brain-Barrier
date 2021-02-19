from imblearn.over_sampling import SMOTE


class Sampler:

    def __init__(self, sampling_strategy):
        self.sampling_strategy = sampling_strategy

    def transform(self, X, y):
        model = SMOTE(sampling_strategy=self.sampling_strategy)
        X_smo, y_smo = model.fit_resample(X, y)
        return X_smo, y_smo

import os
import glob
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from models.feature_cleaner import Selector
from models.oversampler import Sampler
from models.pcr import Transformer, LogClassifier
from models.forest import ForestClassifier

if __name__ == "__main__":

    # with open('parameters.json', 'r') as read_file:
    #
    #     parameters = json.load(read_file)
    #     processed_data = parameters['processed_data']

    data = pd.read_csv('data/processed/substrates.csv')

    # select best features
    selector = Selector(target='substrate',
                        n=50)
    X, y, selected = selector.select(data=data)
    X = X.loc[:, selected.Feature[:20]]

    # make oversampling
    sampler = Sampler(sampling_strategy=0.5)
    X_smo, y_smo = sampler.transform(X, y)

    # split into train/test and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_smo, y_smo, stratify=y_smo)

    # apply PCR
    pca = Transformer(X=X_smo, y=y_smo, n=8)
    transformed = pca.transform()

    log_classifier = LogClassifier(class_weights={0: 1, 1: 5})
    log_scores = log_classifier.cross_validate(X=transformed, y=y_smo, cv=5)

    # apply Random Forest
    forest_classifier = ForestClassifier(class_weight={0: 1, 1: 5},
                                         n_estimators=800,
                                         min_samples_leaf=3)
    rf_scores = forest_classifier.cross_validate(X=X_train, y=y_train, cv=5)
    # print(forest_classifier.validate(X_val, y_val))


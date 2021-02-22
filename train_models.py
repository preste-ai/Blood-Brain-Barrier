import json
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from models.feature_cleaner import Selector
from models.oversampler import Sampler
from models.pcr import Transformer, LogClassifier
from models.forest import ForestClassifier
from utils import plot_relationships


def train_model(dataset, target, train, validate, dump):

    with open('parameters.json', 'r') as read_file:
        parameters = json.load(read_file)

    data = pd.read_csv(parameters['processed_data'] + dataset)

    # select best features
    selector = Selector(target=target,
                        n=parameters['select_with_method'])
    X, y, selected = selector.select(data=data)
    X = X.loc[:, selected.Feature[:parameters['select_overall']]]

    # make oversampling
    sampler = Sampler(sampling_strategy=parameters['sampling_strategy'])
    X_smo, y_smo = sampler.transform(X, y)

    # split into train/test and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_smo, y_smo,
                                                      stratify=y_smo,
                                                      random_state=parameters['random_state'])

    # apply PCR
    pca = Transformer(X=X_smo,
                      y=y_smo,
                      n=parameters['pc_components'])
    transformed = pd.DataFrame(pca.transform())

    log_classifier = LogClassifier(class_weights={int(k): v
                                                  for k, v
                                                  in parameters['class_weights_log'].items()})
    log_scores_train, log_scores_test = log_classifier.cross_validate(X=transformed,
                                                                      y=y_smo,
                                                                      n_splits=parameters['n_splits'])

    # apply Random Forest
    forest_classifier = ForestClassifier(class_weight={int(k): v
                                                       for k, v
                                                       in parameters['class_weights_rf'].items()},
                                         n_estimators=parameters['n_estimators'],
                                         min_samples_leaf=parameters['min_samples_leaf'],
                                         random_state=parameters['random_state'])
    rf_scores_train, rf_scores_test = forest_classifier.cross_validate(X=X_train,
                                                                       y=y_train,
                                                                       n_splits=parameters['n_splits'])
    if train:
        print('PCR scores:\n',
              'Train:\n', log_scores_train,
              '\nTest:\n', log_scores_test)
        print('RF scores:\n',
              'Train:\n', rf_scores_train,
              '\nTest:\n', rf_scores_test)
        return log_scores_train, log_scores_test, rf_scores_train, rf_scores_test

    if validate:
        print(forest_classifier.validate(X_val, y_val))
        return

    if dump:
        dump(forest_classifier, 'models/trained/RFSubstrates.joblib')
        dump(log_classifier, 'models/trained/PCRSubstrates.joblib')

    # plot_relationships(data=pd.concat([X, y], axis=1), target=target)


if __name__ == "__main__":
    log_scores_train, log_scores_test, rf_scores_train, rf_scores_test = \
        train_model(dataset='inhibitors_a2.csv',
                    target='inhibitor',
                    train=True,
                    validate=False,
                    dump=False)

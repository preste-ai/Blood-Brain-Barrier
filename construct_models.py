import json
import glob
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from models.transformers.feature_cleaner import Selector
from models.transformers.oversampler import Sampler
from models.classifiers.pcr import Transformer, LogClassifier
from models.classifiers.forest import ForestClassifier
from utils import plot_relationships


class Constructor:

    def __init__(self, parameters):

        """
        Class with methods to transform dataset, train and validate models
        :param parameters: dictionary with parameters for models
        """

        self.parameters = parameters

    def make_data(self, plot=False):

        """
        Select important features, oversample data, and split data into train and validation sets
        :param plot: boolean, True for plotting features/target relationships
        :return: data set split into train and validation parts
        """

        data = pd.read_csv(self.parameters['dataset_file'])

        # select best features
        selector = Selector(target=self.parameters['target_name'],
                            n=self.parameters['select_with_method'])
        X, y, selected = selector.select(data=data)
        X = X.loc[:, selected.Feature[:self.parameters['select_overall']]]

        # make oversampling
        sampler = Sampler(sampling_strategy=self.parameters['sampling_strategy'],
                          k_neighbors=self.parameters['k_neighbors'],
                          random_state=self.parameters['random_state'])
        X_smo, y_smo = sampler.transform(X, y)

        # split into train/test and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_smo, y_smo,
                                                          stratify=y_smo,
                                                          random_state=self.parameters['random_state'])

        if plot:
            plot_relationships(data=pd.concat([X, y], axis=1),
                               target=self.parameters['target_name'])

        return X_train, X_val, y_train, y_val

    def transform(self, X, y):

        """
        Transform data with PCA
        :param X: array with molecule features and their values
        :param y: array with target values
        :return: fitted PCA transformer and transformed data set
        """

        # apply PCA
        transformer = Transformer(n=self.parameters['pc_components'])
        transformed = pd.DataFrame(transformer.transform(X, y))

        return transformer, transformed

    def train_models(self, X_train, y_train):

        """
        Train Random Forest and PCR classifiers models
        :param X_train: array with molecule features and their values
        :param y_train: array with target values
        :return: scores for train and test data sets for all developed models,
                 trained models: PCA transformer, Logistic Regression model, and Random Forrest Classifier model
        """

        transformer, transformed = self.transform(X=X_train,
                                                  y=y_train)

        log_classifier = LogClassifier(class_weights={int(k): v
                                                      for k, v
                                                      in self.parameters['class_weights_log'].items()})
        log_scores_train, log_scores_test = log_classifier.cross_validate(X=transformed,
                                                                          y=y_train,
                                                                          n_splits=self.parameters['n_splits'])

        # apply Random Forest
        forest_classifier = ForestClassifier(class_weight={int(k): v
                                                           for k, v
                                                           in self.parameters['class_weights_rf'].items()},
                                             n_estimators=self.parameters['n_estimators'],
                                             min_samples_leaf=self.parameters['min_samples_leaf'],
                                             random_state=self.parameters['random_state'])
        rf_scores_train, rf_scores_test = forest_classifier.cross_validate(X=X_train,
                                                                           y=y_train,
                                                                           n_splits=self.parameters['n_splits'])
        scores = {'Log train': log_scores_train.T,
                  'Log test': log_scores_test.T,
                  'RF train': rf_scores_train.T,
                  'RF test': rf_scores_test.T}
        models = transformer, log_classifier, forest_classifier

        return scores, models

    def validate_model(self, model_path, X_val, y_val):

        """
        Validate model
        :param model_path: path to the trained model file
        :param X_val: array with molecule features and their values
        :param y_val: array with target values
        :return: classification report for model predictions on validation set
        """

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        return model.validate(X_val, y_val)

    def run(self, train, dump, validate):

        """
        Method to run Constructor on given data set
        :param train: boolean, True for training model
        :param dump: boolean, True for saving trained model
        :param validate: boolean, True for validating trained model
        :return: depending on train, dump, and validate parameter values:
                 (train=True, dump=False, validate=False) - training/test scores for models
                 (train=True, dump=True, validate=False) - training/test scores for models + save trained models
                 (train=False, dump=False, validate=True) - classification report for trained models predictions
                 on validation set
        """

        X_train, X_val, y_train, y_val = self.make_data()
        _, transformed_val = self.transform(X_val, y_val)

        if train and not dump:
            scores, models = \
                self.train_models(X_train, y_train)
            return scores

        elif train and dump:
            scores, models = \
                self.train_models(X_train, y_train)

            for model in models:

                model_name = type(model).__name__

                if 'Transformer' in model_name:
                    with open(f"{self.parameters['trained_transformers']}/{model_name}.pkl", 'wb') as file:
                        pickle.dump(model, file)
                        continue
                else:
                    with open(f"{self.parameters['trained_models']}/{model_name}.pkl", 'wb') as file:
                        pickle.dump(model, file)
                        continue

            return scores

        elif validate:
            val_scores = {}
            for model_path in glob.glob(self.parameters['trained_models']+'/*.pkl'):
                model_name = model_path.split('/')[-1]
                if 'Log' in model_name:
                    val_scores[model_name] = self.validate_model(model_path=model_path,
                                                                 X_val=transformed_val,
                                                                 y_val=y_val)
                else:
                    val_scores[model_name] = self.validate_model(model_path=model_path,
                                                                 X_val=X_val,
                                                                 y_val=y_val)
            return val_scores


if __name__ == "__main__":

    target = 'inhibitorsB1'
    constructor = Constructor(parameters=json.load(open('parameters.json', 'r'))[target])

    scores = constructor.run(train=True,
                             dump=True,
                             validate=False)

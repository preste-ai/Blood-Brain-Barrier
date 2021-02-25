import json
import glob
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from models.transformers.feature_cleaner import Selector
from models.transformers.oversampler import Sampler
from models.classifiers.pcr import Transformer, LogClassifier
from models.classifiers.forest import ForestClassifier


class Constructor:
    def __init__(self, parameters_file):
        self.parameters = json.load(open(parameters_file, 'r'))

    def make_data(self):

        data = pd.read_csv(self.parameters['processed_data'] + self.parameters['dataset_file'])

        # select best features
        selector = Selector(target=self.parameters['target_name'],
                            n=self.parameters['select_with_method'])
        X, y, selected = selector.select(data=data)
        X = X.loc[:, selected.Feature[:self.parameters['select_overall']]]

        # make oversampling
        sampler = Sampler(sampling_strategy=self.parameters['sampling_strategy'],
                          random_state=self.parameters['random_state'])
        X_smo, y_smo = sampler.transform(X, y)

        # split into train/test and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_smo, y_smo,
                                                          stratify=y_smo,
                                                          random_state=self.parameters['random_state'])

        return X_train, X_val, y_train, y_val

    def transform(self, X, y):

        # apply PCA
        transformer = Transformer(X=X,
                                  y=y,
                                  n=self.parameters['pc_components'])
        transformed = pd.DataFrame(transformer.transform())

        return transformer, transformed

    def train_models(self, X_train, y_train):

        transformer, transformed = self.transform(X=X_train,
                                                  y=y_train)

        log_classifier = LogClassifier(class_weights={int(k): v
                                                      for k, v
                                                      in self.parameters['class_weights_log'].items()},
                                       random_state=self.parameters['random_state'])
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

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        return model.validate(X_val, y_val)

    def run(self, train, validate, dump):

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

                model_name = type(model).__name__ + self.parameters['target_name']

                if 'Transformer' in model_name:
                    with open(f'models/trained/transformers/{model_name}.pkl', 'wb') as file:
                        pickle.dump(model, file)
                        continue
                else:
                    with open(f'models/trained/classifiers/{model_name}.pkl', 'wb') as file:
                        pickle.dump(model, file)
                        continue

            return scores

        elif validate:
            val_scores = {}
            for model_path in glob.glob('models/trained/classifiers/*.pkl'):
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

    substrates = Constructor(parameters_file='parameters_substrates.json')

    train_scores = substrates.run(train=True,
                                  validate=False,
                                  dump=False)

    validation_scores = substrates.run(train=False,
                                       validate=True,
                                       dump=False)
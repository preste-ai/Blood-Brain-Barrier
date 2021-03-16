import glob
import json
import pickle
import pandas as pd
from multiprocessing import freeze_support
from models.transformers.featurizer import MordredFeaturizer

if __name__ == "__main__":

    freeze_support()

    # read parameters file
    with open('parameters.json', 'r') as read_file:
        parameters = json.load(read_file)

    # read data with molecules to predict
    dataset = pd.read_csv(parameters['predictions']['input_data'])
    target = parameters['predictions']['target']

    # describe molecules
    descriptor = MordredFeaturizer(data=dataset)
    described_data = descriptor.described_molecules.drop('Smiles', axis=1)

    # select necessary descriptors for model
    selected_data = described_data.loc[:, parameters[target]['features']]

    # transform data with PCA
    for model_path in glob.glob(parameters[target]['trained_transformers'] + '/*.pkl'):
        with open(model_path, 'rb') as file:
            transformer = pickle.load(file)
            transformed = transformer.model.transform(selected_data)

    # run predicting models
    for model_path in glob.glob(parameters[target]['trained_models'] + '/*.pkl'):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            model_name = type(model).__name__

            # write predictions to file
            if 'Log' in model_name:
                predictions = pd.concat([dataset, pd.DataFrame(model.predict(transformed))], axis=1)
                predictions.to_csv(parameters['predictions']['output_data'].format(model_name))
            else:
                predictions = pd.concat([dataset, pd.DataFrame(model.predict(selected_data))], axis=1)
                predictions.to_csv(parameters['predictions']['output_data'].format(model_name))

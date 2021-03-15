import glob
import json
import pickle
import pandas as pd
from multiprocessing import freeze_support
from models.transformers.featurizer import MordredFeaturizer

if __name__ == "__main__":

    freeze_support()

    dataset = pd.read_csv('data/predictions/test.csv')
    target = 'substrates'

    with open('parameters.json', 'r') as read_file:
        parameters = json.load(read_file)[target]

    descriptor = MordredFeaturizer(data=dataset)
    described_data = descriptor.described_molecules.drop('Smiles', axis=1)

    selected_data = described_data.loc[:, parameters['features']]

    for model_path in glob.glob(parameters['trained_transformers'] + '/*.pkl'):
        with open(model_path, 'rb') as file:
            transformer = pickle.load(file)
            transformed = transformer.model.transform(selected_data)

    for model_path in glob.glob(parameters['trained_models'] + '/*.pkl'):
        with open(model_path, 'rb') as file:

            model = pickle.load(file)
            model_name = type(model).__name__

            if 'Log' in model_name:
                predictions = pd.concat([dataset, pd.DataFrame(model.predict(transformed))], axis=1)
                predictions.to_csv(f'data/predictions/predictions{model_name}.csv')
            else:
                predictions = pd.concat([dataset, pd.DataFrame(model.predict(selected_data))], axis=1)
                predictions.to_csv(f'data/predictions/predictions{model_name}.csv')

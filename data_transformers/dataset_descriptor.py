import os
import glob
import json
import pandas as pd
from multiprocessing import freeze_support
from models.transformers.featurizer import MordredFeaturizer

if __name__ == "__main__":

    freeze_support()

    with open('parameters.json', 'r') as read_file:

        parameters = json.load(read_file)['general']

    for file in glob.glob(parameters['raw_data']+'*.csv'):

        dataset = pd.read_csv(file)
        filename = os.path.basename(file)

        transformer = MordredFeaturizer(data=dataset)

        described_data = transformer.described_molecules

        described_data.to_csv(parameters['described_data'] + filename)

import os
import glob
import json
import pandas as pd
from multiprocessing import freeze_support
from models.transformers.featurizer import MordredFeaturizer

if __name__ == "__main__":

    with open('parameters.json', 'r') as read_file:

        parameters = json.load(read_file)['general']

    freeze_support()

    for file in glob.glob(parameters['raw_data']+'*.csv'):

        dataset = pd.read_csv(file)
        filename = os.path.basename(file)

        transformer = MordredFeaturizer(data=dataset,
                                        calculator=None,
                                        described_molecules=None)

        described_data = transformer.featurize()

        described_data.to_csv(parameters['described_data'] + filename)

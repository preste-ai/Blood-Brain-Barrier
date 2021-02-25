import os
import glob
import json
import pandas as pd
from multiprocessing import freeze_support
from models.transformers.featurizer import MordredFeaturizer

if __name__ == "__main__":

    with open('../parameters_substrates.json', 'r') as read_file:

        parameters = json.load(read_file)
        raw_data = parameters['raw_data']

    freeze_support()

    for file in glob.glob(raw_data):

        dataset = pd.read_csv(file)
        filename = os.path.basename(file)

        transformer = MordredFeaturizer(data=dataset,
                                        calculator=None,
                                        described_molecules=None)

        described_data = transformer.featurize()

        described_data.to_csv('data/described/' + filename)

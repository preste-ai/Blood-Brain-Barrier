import os
import glob
import json
import pandas as pd

if __name__ == "__main__":

    with open('parameters.json', 'r') as read_file:

        parameters = json.load(read_file)['general']
        described_data = parameters['described_data']

    # put described molecules from datasets in three different categories
    datasets = {'substrates': [],
                'inhibitors_a2': [],
                'inhibitors_b1': []}

    for file in glob.glob(described_data + '*.csv'):

        filename = os.path.basename(file)
        dataset = pd.read_csv(file, index_col=0)

        if 'substrates' in filename:
            datasets['substrates'].append(dataset)
        else:
            if 'A2' in filename:
                datasets['inhibitors_a2'].append(dataset)
            else:
                datasets['inhibitors_b1'].append(dataset)

    # concatenate data sets from categories
    datasets['substrates'] = pd.concat(
        datasets['substrates'] +
        datasets['inhibitors_a2'] +
        datasets['inhibitors_b1'])
    datasets['substrates'].drop(['inhibitor', 'Standard Value'], axis=1, inplace=True)

    datasets['inhibitors_a2'] = pd.concat(datasets['inhibitors_a2'])
    datasets['inhibitors_b1'] = pd.concat(datasets['inhibitors_b1'])

    # do little cleanup, write data sets to files
    for name, dataset in datasets.items():

        dataset = dataset.\
            dropna(subset=['Smiles']).\
            drop_duplicates(subset='Smiles').\
            set_index('Molecule ChEMBL ID')

        dataset.to_csv(parameters['processed_data'] + f'{name}.csv')

import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors


class MordredFeaturizer:

    def __init__(self, data):

        """
        Class for molecules featurization with Mordred
        :param data: DataFrame with molecules SMILES to featurize
        """

        self.data = data
        self.calculator = Calculator(descriptors, ignore_3D=True)
        self.described_molecules = self.featurize()

    def make_molecule(self):

        """
        Make RDKit molecules objects from SMILES
        :return:
        """

        self.data.molecules = self.data.Smiles.apply(Chem.MolFromSmiles)

        return self.data.molecules

    def featurize(self):

        """
        Featurize molecules with Mordred descriptors
        :return: featurized data set
        """

        self.make_molecule()

        self.described_molecules = self.calculator.pandas(self.data.molecules)
        self.described_molecules.index = self.data.index

        self.described_molecules = pd.concat([self.data, self.described_molecules], axis=1)

        return self.described_molecules

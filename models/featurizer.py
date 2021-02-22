import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors


class MordredFeaturizer:

    def __init__(self, calculator, data, described_molecules):

        """

        :param calculator:
        :param data:
        :param described_molecules:
        """

        self.calculator = calculator
        self.data = data
        self.described_molecules = described_molecules

    def make_molecule(self):

        """

        :return:
        """

        self.data.molecules = self.data.Smiles.apply(Chem.MolFromSmiles)

        return self.data.molecules

    def featurize(self):

        """

        :return:
        """

        if self.calculator is None:
            self.calculator = Calculator(descriptors, ignore_3D=True)

        self.make_molecule()

        self.described_molecules = self.calculator.pandas(self.data.molecules)
        self.described_molecules.index = self.data.index

        self.described_molecules = pd.concat([self.data, self.described_molecules], axis=1)

        return self.described_molecules

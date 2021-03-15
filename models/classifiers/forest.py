from sklearn.ensemble import RandomForestClassifier
from models.classifiers.base import BaseClassifier


class ForestClassifier(BaseClassifier):

    def __init__(self, class_weight, n_estimators, min_samples_leaf, random_state):

        """
        Forrest classifier model class
        :param class_weight: dictionary with weights for different classes
        :param n_estimators: number of trees for the Forrest Classifier
        :param min_samples_leaf: minimal number of samples in leaves in each tree
        :param random_state: number to control randomness
        """

        super(BaseClassifier, self).__init__()
        self.class_weight = class_weight
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = RandomForestClassifier(class_weight=self.class_weight,
                                            n_estimators=self.n_estimators,
                                            min_samples_leaf=self.min_samples_leaf,
                                            random_state=self.random_state)

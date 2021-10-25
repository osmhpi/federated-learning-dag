import numpy as np
import sys

from .node import Node, NodeConfiguration
from .poison_type import PoisonType

FLIP_FROM_CLASS = 3
FLIP_TO_CLASS = 8

class MaliciousNode(Node):
    def __init__(self, tangle, tx_store, tip_selector, client_id, cluster_id, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None, poison_type=PoisonType.Disabled, config=NodeConfiguration()):
        self.poison_type = poison_type

        if self.poison_type == PoisonType.LabelFlip:
            def flip_labels(dataset):
                dataset = {'x': dataset['x'], 'y': np.copy(dataset['y'])}

                flip_from_indices = [i for i, label in enumerate(dataset['y']) if label == FLIP_FROM_CLASS]
                flip_to_indices = [i for i, label in enumerate(dataset['y']) if label == FLIP_TO_CLASS]

                for i in flip_from_indices:
                    dataset['y'][i] = FLIP_TO_CLASS

                for i in flip_to_indices:
                    dataset['y'][i] = FLIP_FROM_CLASS

                return dataset

            train_data = flip_labels(train_data)
            eval_data = flip_labels(eval_data)

        super().__init__(tangle, tx_store, tip_selector, client_id, cluster_id, train_data, eval_data, model=model, config=config)

    def train(self, model_params):
        if self.poison_type == PoisonType.Random:
            malicious_weights = [np.random.RandomState().normal(size=w.shape) for w in model_params]
            return malicious_weights
        else:
            return super().train(model_params)


    def create_transaction(self):
        t, weights = super().create_transaction()

        if t is not None and self.poison_type != PoisonType.Disabled:
            t.add_metadata('poisoned', True)

        return t, weights

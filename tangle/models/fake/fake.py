import numpy as np
from numpy.random import rand
import random
from ..model import Model

class ClientModel(Model):

    def __init__(self, seed):
        # Is close enough to weight_size of femnist model
        self._parameters = rand(3300000)

    def set_params(self, model_params):
        self._parameters = model_params

    def get_params(self):
        return self._parameters

    def train(self, data, num_epochs=1, batch_size=10):
        self._parameters = rand(3300000)
        return None

    def test(self, data):
        return {
            "loss": random.random(),
            "accuracy": random.random()
        }

    # 'Implement' abstract methods (won't be called)
    def create_model(self):
        return None, None, None, None, None, None

    def process_x(self, raw_x_batch):
        pass

    def process_y(self, raw_y_batch):
        pass

import ray
import numpy as np

from ..lab import Dataset

class RayDataset(Dataset):
    def __init__(self, lab_config, model_config):
        super().__init__(lab_config, model_config)

        self.remote_train_data = {
            cid : { 'x': ray.put(np.asarray(data['x'])), 'y': ray.put(np.asarray(data['y'])) } for (cid, data) in self.train_data.items()
        }
        self.remote_test_data = {
            cid : { 'x': ray.put(np.asarray(data['x'])), 'y': ray.put(np.asarray(data['y'])) } for (cid, data) in self.test_data.items()
        }

        self.train_data = None
        self.test_data = None

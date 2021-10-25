import os
import itertools
import json
import numpy as np
from collections import defaultdict

class Dataset:
    def __init__(self, lab_config, model_config):
        self.lab_config = lab_config
        self.model_config = model_config
        self.clients, self.train_data, self.test_data = self.setup_clients(model_config.limit_clients)

    def setup_clients(self, limit_clients):
        eval_set = 'test' if not self.lab_config.use_val_set else 'val'
        train_data_dir = os.path.join(self.lab_config.model_data_dir, 'train')
        test_data_dir = os.path.join(self.lab_config.model_data_dir, eval_set)

        users, cluster_ids, train_data, test_data = read_data(train_data_dir, test_data_dir)

        clients = list(itertools.zip_longest(users, cluster_ids))

        if (limit_clients > 0):
            rng_state = np.random.get_state()
            np.random.seed(4711)
            clients = [clients[i] for i in np.random.choice(len(clients), limit_clients, replace=False)]
            np.random.set_state(rng_state)

        return clients, train_data, test_data

    def select_clients(self, my_round, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Returns:
            list of (client_id, cluster_id)
        """
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(my_round)
        client_indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return [self.clients[i] for i in client_indices]


def batch_data(data, batch_size, num_batches, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''

    data_repetitions = (batch_size / len(data['x'])) * num_batches
    print(f'batch_data will return {data_repetitions} times the data')
    data_repetitions = np.ceil(data_repetitions)
    data_repetitions = max(data_repetitions, 1)

    data_x = np.repeat(data['x'], data_repetitions, axis=0)
    data_y = np.repeat(data['y'], data_repetitions, axis=0)

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    data_x = np.random.permutation(data_x)
    np.random.set_state(rng_state)
    data_y = np.random.permutation(data_y)


    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        if i >= num_batches * batch_size:
            break
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def read_dir(data_dir):
    clients = []
    cluster_ids = {}
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'cluster_ids' in cdata:
            for idx, u in enumerate(cdata['users']):
                cluster_ids[u] = cdata['cluster_ids'][idx]
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    # If there are no cluser_ids in the data, assign 0 for each user
    cluster_ids = [cluster_ids[c] if c in cluster_ids else 0 for c in clients]
    return clients, cluster_ids, groups, data

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users', 'user_data' and 'cluster_ids'
    - the set of train set users is the same as the set of test set users

    Returns:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    print("Reading Data...")
    train_clients, train_cluster_ids, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_cluster_ids, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups
    assert train_cluster_ids == test_cluster_ids

    print("Done Reading Data...")
    # Todo return groups if required
    return train_clients, train_cluster_ids, train_data, test_data

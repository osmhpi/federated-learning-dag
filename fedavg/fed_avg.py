import sys
import time
import ray
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tangle.lab import Lab
from tangle.core import Node

class DummyTipSelector():
    def compute_ratings(self, node):
        pass

############## Training ##############

@ray.remote
def train_single(client_id, data, model_config, global_params, seed):
    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    model = Lab.create_client_model(seed, model_config)

    data = {'x': ray.get(data['x']), 'y': ray.get(data['y'])}

    node = Node(None, None, DummyTipSelector(), client_id, None, data, None, model)
    new_params = node.train(global_params)

    global_params = np.array(global_params)
    new_params = np.array(new_params)

    return global_params - new_params, len(data['y'])


def train(dataset, run_config, model_config, seed, lr=1.):
    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model = Lab.create_client_model(seed, model_config)
    global_params = model.get_params()

    accuracies = []
    unique_cids = get_unique_cluster_ids(dataset.clients)

    for epoch in range(run_config.num_rounds):
        print("Started training for round %d" % epoch)
        start = time.time()
        clients = dataset.select_clients(epoch, run_config.clients_per_round)

        futures = [train_single.remote(client_id, dataset.remote_train_data[client_id], model_config, global_params, seed) for (client_id, cluster_id) in clients]

        param_update = 0
        total_weight = 0
        for param_diff, weight in ray.get(futures):
            param_update += param_diff * weight
            total_weight += weight

        param_update /= total_weight

        global_params -= lr * param_update

        end_training = time.time()
        print('Time for epoch {} is {} sec'.format(epoch, end_training - start))

        # maybe add an Option to deactivate this?
        accuracies.append(test_acc_per_cluser(global_params, dataset, model_config, clients, seed, unique_cids))
        print('Time for testing clients of epoch {} is {} sec'.format(epoch, time.time() - end_training))

        if run_config.eval_every != -1 and epoch % run_config.eval_every == 0:
            mean_accuracy = test_mean_acc_eval_fraction(global_params, dataset, model_config, run_config.eval_on_fraction, seed)
            print(f'Test Accuracy on {int(run_config.eval_on_fraction * len(dataset.clients))} clients: {mean_accuracy}')
        sys.stdout.flush()

    # compute average test error
    mean_accuracy = test_mean_acc_eval_fraction(global_params, dataset, model_config, 1, seed)
    print(f'Test Accuracy on all Clients: {mean_accuracy}')

    plot_accuracy_boxplot(accuracies, unique_cids)


############## Testing ##############

def test_acc_per_cluser(global_params, dataset, model_config, clients_to_test_on, seed, cids):
    accuracies_per_cluster = {}

    for cid in cids:
        accuracies_per_cluster[cid] = []
    
    accuracies = _test_acc_clients(global_params, dataset, model_config, clients_to_test_on, seed)

    for idx, (_, cid) in enumerate(clients_to_test_on):
        accuracies_per_cluster[cid].append(accuracies[idx])
    
    return accuracies_per_cluster

def test_mean_acc_eval_fraction(global_params, dataset, model_config, eval_on_fraction, seed):
    client_indices = np.random.choice(range(len(dataset.clients)),
                                      min(int(len(dataset.clients) * eval_on_fraction), len(dataset.clients)),
                                      replace=False)
    validation_clients = [dataset.clients[i] for i in client_indices]
    
    accuracies = _test_acc_clients(global_params, dataset, model_config, validation_clients, seed)
    return np.mean(accuracies)

def _test_acc_clients(global_params, dataset, model_config, clients_to_test_on, seed):
    futures = [_test_single.remote(client_id, global_params, dataset.remote_test_data[client_id], model_config, seed)
               for (client_id, _) in clients_to_test_on]
    
    return ray.get(futures)

@ray.remote
def _test_single(client_id, global_params, test_data, model_config, seed):
    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model = Lab.create_client_model(seed, model_config)
    test_data = {'x': ray.get(test_data['x']), 'y': ray.get(test_data['y'])}
    node = Node(None, None, DummyTipSelector(), client_id, None, None, test_data, model)

    results = node.test(global_params, 'test')

    return results['accuracy']


def test(global_params, dataset, model_config, eval_on_fraction, seed):
    client_indices = np.random.choice(range(len(dataset.clients)),
                                      max(min(int(len(dataset.clients) * eval_on_fraction), len(dataset.clients)), 1),
                                      replace=False)
    print(f'testing on {len(client_indices)} clients')
    validation_clients = [dataset.clients[i] for i in client_indices]
    futures = [test_single.remote(client_id, global_params, dataset.remote_test_data[client_id], model_config, seed)
               for (client_id, cluster_id) in validation_clients]

    return np.mean(ray.get(futures))

############## Helpers ##############

def get_unique_cluster_ids(clients):
    return list({ cid for (_, cid) in clients })

def plot_accuracy_boxplot(data, cids, print_avg_acc=False):
    # print for each cluster
    for cid in cids:
        cluster_data = [epoch[cid] for epoch in data if cid in epoch]
        _plot_accuracy_boxplot(cluster_data, cid, print_avg_acc)

    # print for all clusters
    all_cluster_data = [sum(epoch.values(), []) for epoch in data]
    print(all_cluster_data)
    with open('fed_avg_accuracy_per_round_all.txt', 'w') as f:
        for round_number, round_data in enumerate(all_cluster_data):
            f.write(f'{round_number+1} {" ".join(map(str,round_data))}\n')

    _plot_accuracy_boxplot(all_cluster_data, "all", print_avg_acc)

def _plot_accuracy_boxplot(data, cid, print_avg_acc, max_y=1):
    last_generation = len(data)

    plt.boxplot(data)

    # Fix y axis data range to [0, 1]
    plt.ylim([0, max_y])

    if print_avg_acc:
        plt.plot([i for i in range(last_generation)], [np.mean(x) for x in data])
    
    # Settings for plot
    plt.title("Accuracy per round (cluster: %s)" % cid)
    
    plt.xlabel("")
    plt.xticks([i for i in range(last_generation)], [i if i % 10 == 0 else '' for i in range(last_generation)])
    
    plt.ylabel("")
    
    analysis_filepath = ("fed_avg_accuracy_per_round_cluster_%s" % cid)
    plt.savefig(analysis_filepath+".png")

    plt.title("")
    plt.savefig(analysis_filepath+".pdf")
    
    plt.clf()
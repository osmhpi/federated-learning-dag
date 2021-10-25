import copy
import json
import numpy as np
import os
import random
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

from ..lab.dataset import read_data

from sknetwork.clustering import Louvain, modularity

def load(tangle_json_path):
    with open(tangle_json_path) as inf:
        tangle_data = json.load(inf)

    nid_to_client = {}
    clients_to_clusters = {}

    for n in tangle_data['nodes']:
        if 'issuer' in n['metadata']:
            nid_to_client[n['id']] = n['metadata']['issuer']

            if 'clusterId' in n['metadata']:
                clients_to_clusters[n['metadata']['issuer']] = n['metadata']['clusterId']
        else:
            nid_to_client[n['id']] = 'genesis'

    clients = list(set(nid_to_client.values()))

    client_to_idx = {}

    for idx, c in enumerate(clients):
        client_to_idx[c] = idx
    client_to_idx["genesis"] = -1

    def nid_to_idx(nid):
        return client_to_idx[nid_to_client[nid]]

    approval_count = np.zeros((len(clients), len(clients)))

    for n in tangle_data['nodes']:
        # skip genesis round
        if len(n['parents']) == 0:
            continue

        for p in n['parents']:
            if nid_to_client[p] == "genesis":
                continue

            # no self loops
            if nid_to_idx(n['id']) == nid_to_idx(p):
                continue

            approval_count[nid_to_idx(n['id'])][nid_to_idx(p)] += 1
            # approval_count[nid_to_idx(p)][nid_to_idx(n['id'])] += 1

    idx_to_client = clients

    return approval_count, idx_to_client, clients_to_clusters

def compute_clusters(approval_count):
    louvain = Louvain(modularity='newman')
    adjacency = approval_count
    labels = louvain.fit_transform(adjacency)
    return labels, modularity(approval_count, labels)

def partitions(labels, idx_to_client, num_clusters):
    clusters = [[] for z in range(num_clusters)]
    for idx, label in enumerate(labels):
        if label < num_clusters:
            t = idx_to_client[idx]
            clusters[label].append(t)

    return clusters

def load_dataset(d):
    _, _, _, test_data = read_data(d + '/train', d + '/test')
    return test_data

def sample_labels(partitions, dataset, label, known_labels):
    plt.figure(figsize=(20,10))
    num_samples = 5
    columns = len(partitions)
    for i, p in enumerate(partitions):
        s = 0
        for c in np.random.permutation(p):
            client_data = dataset[c]
            if client_data is None:
                # c == 'genesis'
                continue
            client_samples = client_data['x']
            client_labels = [known_labels[x] for x in client_data['y']]
            try:
                sample_idx = client_labels.index(label)
                plt.subplot(num_samples + 1, columns, (s * columns) + i + 1)
                plt.imshow(np.reshape(client_samples[sample_idx], (-1, 28)))
                s += 1
                if s >= num_samples:
                    break
            except:
                pass

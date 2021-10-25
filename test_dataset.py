import numpy as np
import argparse
from tangle.lab import Dataset
from tangle.lab.config.lab_configuration import LabConfiguration


def main(data_dir, num_clusters, num_classes=10):
    labConfig = LabConfiguration()
    labConfig.model_data_dir = './data/' + data_dir

    dataset = Dataset(labConfig, None)

    cluster_clients = {}
    for i in range(num_clusters):
        cluster_clients[i] = []
    for (client, cluster_id) in dataset.clients:
        cluster_clients[cluster_id].append(client)
    print(f'Number of Clients: {len(dataset.clients)}, cluster distribution: {[len(cluster_clients[i]) for i in range(num_clusters)]}')

    cluster_train_data = {cluster: np.concatenate([dataset.train_data[client]['y'] for client in clients]) for (cluster, clients) in cluster_clients.items()}
    print('Train data:')
    for cluster in range(num_clusters):
        hist, _ = np.histogram(cluster_train_data[cluster], bins=range(num_classes+1))
        print(hist)

    for cluster in range(num_clusters):
        dataset_sizes = [len(dataset.train_data[client]['y']) for client in cluster_clients[cluster]]
        mean_size = np.mean(dataset_sizes)
        std_size = np.std(dataset_sizes)
        print(f'Cluster {cluster}: data size per client: {mean_size} +/- {std_size}')

    cluster_test_data = {cluster: np.concatenate([dataset.test_data[client]['y'] for client in clients]) for (cluster, clients) in cluster_clients.items()}
    print('\nTest data:')
    for cluster in range(num_clusters):
        hist, _ = np.histogram(cluster_test_data[cluster], bins=range(num_classes+1))
        print(hist)

    for cluster in range(num_clusters):
        dataset_sizes = [len(dataset.test_data[client]['y']) for client in cluster_clients[cluster]]
        mean_size = np.mean(dataset_sizes)
        std_size = np.std(dataset_sizes)
        print(f'Cluster {cluster}: data size per client: {mean_size} +/- {std_size}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        help='dir for data ./data/x',
                        type=str,
                        required=True)
    parser.add_argument('--num-clusters',
                        help='number of clusters',
                        type=int,
                        default=3)
    parser.add_argument('--num-classes',
                        help='number of classes',
                        type=int,
                        default=10)
    args = parser.parse_args()
    main(args.data_dir, args.num_clusters, args.num_classes)


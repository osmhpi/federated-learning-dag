
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.decomposition import PCA

def preprocessData(data):
    pca = PCA(n_components=2)
    
    labels = []
    cluster_ids = []

    users = data['users']
    user_data = data['user_data']

    dim = len(user_data[users[0]]['x'][0])
    all_data = np.empty(shape=(0, dim))

    for user in users:
        all_data = np.concatenate((all_data, np.array(user_data[user]['x'])), axis=0)
        user_labels = user_data[user]['y']
        labels.extend(user_labels)
    
    for i in range(len(users)):
        for j in range(data['num_samples'][i]):
            cluster_ids.append(data['cluster_ids'][i])
    
    projected_data = pca.fit_transform(all_data)

    return projected_data, labels, cluster_ids

def presentData(args):
    points_x, points_y, labels, cluster_ids = loadData(args.filepath)
    print("Example data has following labels:", set(labels))

    if args.highlight == 'CLUSTER':
        data_to_present = cluster_ids
    elif args.highlight == 'LABEL':
        data_to_present = labels
    
    plt.scatter(points_x, points_y, c=data_to_present, s=3)
    plt.show()

def loadData(filepath: str):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_dir = os.path.join(current_dir, *filepath.split('/'))

    with open(file_dir) as inf:
        data = json.load(inf)

    users = data['users']
    user_data = data['user_data']

    projected_data, labels, cluster_ids = preprocessData(data)
    points_x, points_y = splitData(projected_data)

    return points_x, points_y, labels, cluster_ids

def splitData(data):
    x_data = [x[0] for x in data]
    y_data = [x[1] for x in data]
    return x_data, y_data

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--filepath',
        help='Specify which data file should be shown. Default: "data/all_data/data.json"',
        default='data/all_data/data.json',
        type=str,
        required=False)
    
    parser.add_argument(
        '--highlight',
        help='Color assignments according to clustersor labels. Default: label',
        type=str,
        choices=['LABEL', 'CLUSTER'],
        default='LABEL',
        required=False)
    
    return parser.parse_args()

args = parse_args()
presentData(args)

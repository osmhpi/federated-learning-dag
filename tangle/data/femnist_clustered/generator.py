import json
import math
import os
import pydash
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import defaultdict

pd.set_option('display.max_columns', 500)

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
            print("hierarchies exist")
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def clean_and_split(data, test_split=0.1):
    cleaned = {}
    cleaned_test = {}
    cluster_ids = [None] * len(users)
    for user_index, username in enumerate(users):
        userdata = {}
        userdata['x']= []
        userdata['y']= []
        for index, y in enumerate(data[username]['y']):
            if y < 10 and y in [user_index % 5 , user_index % 5 + 5]:
                userdata['x'].append(data[username]['x'][index])
                userdata['y'].append(0 if y<5 else 1)
        if (len(userdata['y']) > 0):
            train_size = math.floor(len(userdata['y']) * (1 - test_split))
            cleaned[username]= {'x': userdata['x'][:train_size],
                                'y': userdata['y'][:train_size]}
            cleaned_test[username]= {'x': userdata['x'][train_size:],
                                     'y': userdata['y'][train_size:]}
        else:
            print('Not enough data for client {}'.format(username))
        cluster_ids[user_index] = user_index % 5
    return cleaned, cleaned_test, cluster_ids

users, _, data,  = read_dir('../../.././data/femnist-data/large/train')
_, _, data_test,  = read_dir('../../.././data/femnist-data/large/test')

complete_data = data.copy()
for username in users:
    complete_data[username]['x'].extend(data_test[username]['x'])
    complete_data[username]['y'].extend(data_test[username]['y'])

train_output = {}
test_output = {}
train_output['user_data'], test_output['user_data'], train_output['cluster_ids'] = clean_and_split(complete_data)
train_output['users'] = list(train_output['user_data'].keys())

test_output['cluster_ids'] = train_output['cluster_ids']
test_output['users'] = train_output['users']
with open('data/train/data.json', 'w') as file:
    json.dump(train_output, file)
with open('data/test/data.json', 'w') as file:
    json.dump(test_output, file)

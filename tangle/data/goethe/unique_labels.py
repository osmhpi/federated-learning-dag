import json
import numpy as np
import os
import string

def print_unique_y():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, "data", "train")

    unique_y = set()

    for subdir, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".json"):
                print("Processing %s now." % filepath)
                y_set = load_data(filepath)

                unique_y = unique_y.union(y_set)
    
    print(sorted(unique_y))
    print(len(unique_y))
    print('é' in unique_y)

def load_data(filepath):
    with open(filepath) as inf:
        data = json.load(inf)
    
    y_set = set()

    users = data['users']
    user_data = data['user_data']

    for user in users:
        y_set = y_set.union(set(user_data[user]['y']))

    return y_set

print_unique_y()

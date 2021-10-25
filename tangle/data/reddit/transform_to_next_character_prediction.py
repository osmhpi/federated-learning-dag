import argparse
import errno
import json
import numpy as np
import os
import string

def transform_to_next_character_prediction(args):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, args.datadir)
    output_dir = os.path.join(current_dir, args.outputdir)
    min_data_entries = args.min_data_entries

    for subdir, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = subdir + os.sep + filename
            output_path = output_dir + subdir[len(data_dir):] + os.sep + filename

            if filepath.endswith(".json"):
                users, user_data = load_data(filepath)

                # Will be the same like users, but without duplicates
                cleaned_users = []
                # Track processed users to find duplictaes
                processed_users = set()
                # Number of samples per user
                num_samples = []
                # Cluster id per user
                cluster_ids = []
                
                for user in users:
                    # At least one user (4482) can be found twice in users
                    if (user not in processed_users):
                        data_x, data_y = to_next_character_prediction(user_data[user])
                        processed_users.add(user)
                        samples_of_user = len(data_y)
                        # Only keep users, that have at least min entries
                        if samples_of_user >= min_data_entries:
                            user_data[user]['x'] = data_x
                            user_data[user]['y'] = data_y
                            cleaned_users.append(user)
                            num_samples.append(samples_of_user)
                            cluster_ids.append(args.cluster_id)
                        # Remove users with less than the specified min entries
                        else:
                            user_data.pop(user)

                assert(len(cleaned_users) == len(num_samples))
                assert(len(cleaned_users) == len(cluster_ids))
                assert(len(cleaned_users) == len(user_data))

                save_data(cleaned_users, num_samples, cluster_ids, user_data, output_path)


def to_next_character_prediction(user_data, seq_length=80):
    """Converts reddit natural language processing format into next character prediction as in shakespeare.

    Args:
        user_data: user_data of one specific user
        seq_length: length of strings in X
    """

    # Also remove ", because shakespeare data does not use "
    TOKENS_TO_REMOVE = ['<BOS>', '<EOS>', '<PAD>', '"']
    
    raw_text = []
    data_y = user_data['y']

    # Reconstruct the possible original text
    raw_data = [np.array(x['target_tokens'], dtype=str).flatten() for x in data_y]
    for x in raw_data:
        x = [x for x in x if x not in TOKENS_TO_REMOVE]
        raw_text.extend(x)
    raw_text = ''.join([' ' + x if x not in string.punctuation else x for x in raw_text])
    raw_text = raw_text.strip()

    # Preprocess the "original" text to fit next character format
    dataX = []
    dataY = []
    for i in range(0, len(raw_text) - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append(seq_in)
        dataY.append(seq_out)
    return dataX, dataY

def load_data(filepath):
    with open(filepath) as inf:
        data = json.load(inf)

    users = data['users']
    user_data = data['user_data']

    return users, user_data

def save_data(users, num_samples, cluster_ids, user_data, filepath):
    all_data = {}
    all_data['users'] = users
    all_data['num_samples'] = num_samples
    all_data['cluster_ids'] = cluster_ids
    all_data['user_data'] = user_data
    
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filepath, 'w') as outfile:
        json.dump(all_data, outfile)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--datadir',
        help='Specify where to search for reddit data. Default: "data"',
        default='data',
        type=str,
        required=False)

    parser.add_argument(
        '--outputdir',
        help='Specify where to search for reddit data. Default: "data-transformed"',
        default='data-transformed',
        type=str,
        required=False)

    parser.add_argument(
        '--cluster-id',
        help='Specify the clusterId, all data will be assigned to. Default: 1',
        default=1,
        type=int,
        required=False)

    parser.add_argument(
        '--min-data-entries',
        help='Specify the minimum number of data entries, a user needs to have. Default: 5',
        default=5,
        type=int,
        required=False)

    return parser.parse_args()

args = parse_args()
transform_to_next_character_prediction(args)

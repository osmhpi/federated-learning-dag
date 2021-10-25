import argparse
import json
import numpy as np
import os

import data_generator as generator

def main():
	args = parse_args()
	np.random.seed(args.seed)

	print('Generating dataset')
	num_samples = get_num_samples(args.num_tasks)
	dataset = generator.SyntheticDataset(
		num_classes=args.num_classes, prob_clusters=args.prob_clusters, num_dim=args.num_dim, seed=args.seed)
	tasks = [dataset.get_task(s) for s in num_samples]
	users, num_samples, user_data, cluster_ids = to_leaf_format(tasks)
	save_json('data/all_data', 'data.json', users, num_samples, user_data, cluster_ids)
	print('Done :D')


def get_num_samples(num_tasks, min_num_samples=5, max_num_samples=1000):
	num_samples = np.random.lognormal(3, 2, (num_tasks)).astype(int)
	num_samples = [min(s + min_num_samples, max_num_samples) for s in num_samples]
	return num_samples


def to_leaf_format(tasks):
	users, num_samples, user_data, cluster_ids = [], [], {}, []
	
	for i, t in enumerate(tasks):
		x, y = t['x'].tolist(), t['y'].tolist()
		u_id = str(i)

		users.append(u_id)
		num_samples.append(len(y))
		cluster_ids.append(int(t['cluster']))
		user_data[u_id] = {'x': x, 'y': y}

	return users, num_samples, user_data, cluster_ids


def save_json(json_dir, json_name, users, num_samples, user_data, cluster_ids):
	if not os.path.exists(json_dir):
		os.makedirs(json_dir)
	
	json_file = {
		'users': users,
		'num_samples': num_samples,
		'user_data': user_data,
		'cluster_ids': cluster_ids,
	}
	
	with open(os.path.join(json_dir, json_name), 'w') as outfile:
		json.dump(json_file, outfile)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-num-tasks',
		help='number of devices;',
		type=int,
		required=True)
	parser.add_argument(
		'-num-classes',
		help='number of classes;',
		type=int,
		required=True)
	parser.add_argument(
		'-num-dim',
		help='number of dimensions;',
		type=int,
		required=True)
	parser.add_argument(
		'-seed',
		help='seed for the random processes;',
		type=int,
		default=931231,
		required=False)
	parser.add_argument(
		'--prob-clusters',
		help='probability distribution for clusters;',
		nargs="*",
		type=float,
		default=[0.5,0.5],
		)
	return parser.parse_args()


if __name__ == '__main__':
	main()

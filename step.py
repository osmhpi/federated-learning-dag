#! /usr/bin/env python

import itertools
import os
import sys
from shutil import copy, rmtree
from distutils.dir_util import copy_tree

from leaf.models.poison_type import PoisonType

sys.path.insert(1, './leaf/models')

import numpy as np
import tensorflow as tf

from tangle import Tangle, train_single, AccuracyTipSelectorSettings, TipSelectorIdentifiers
from utils.model_utils import read_data
from utils.args import parse_args

def main():

    args = parse_args()

    ###### Parameters ######
    experiment_name = 'femnist-cnn-0'
    config = 0
    last_generation = 80     # The generation from where to start
    num_clients = 10         # The number of clients used per round

    tip_selector_to_use = TipSelectorIdentifiers.ACCURACY_TIP_SELECTOR

    client_id = 'f3478_49'
    cluster_id = '2'        # Arbitrary value, as it has no effect on the calculation, nor will it be in the output
    ########################

    tip_selection_settings = { 'tip_selector_to_use': tip_selector_to_use,
                               AccuracyTipSelectorSettings.SELECTION_STRATEGY: 'WALK',
                               AccuracyTipSelectorSettings.CUMULATE_RATINGS: False,
                               AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT: 'EXPONENTIAL',
                               AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS: 'WEIGHTED_CHOICE',
                               AccuracyTipSelectorSettings.ALPHA: 0.01 }

    train_data_dir = os.path.join('leaf', 'data', args.dataset, 'data', 'train')
    test_data_dir = os.path.join('leaf', 'data', args.dataset, 'data', 'test')

    print("Loading data...")
    users, cluster_ids, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    print("Loading data... complete")

    # Change into result folder within the experiment folder
    result_path = os.path.join('./experiments', experiment_name, 'config_%s' % config)
    os.chdir(result_path)

    # To execute the step add leaf framework to path
    models_path = os.path.join(sys.path[0], 'leaf', 'models')
    sys.path.insert(1, models_path)

    # Perform the step
    tangle_name = '%s_clients_%s' % (num_clients, last_generation)
    print(train_single(client_id, cluster_id, last_generation + 1, 0, 0, train_data[client_id], test_data[client_id], tangle_name, False, PoisonType.NONE, tip_selection_settings))

if __name__ == '__main__':
    main()

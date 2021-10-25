import os
import random
import sys
import time

import numpy as np
import importlib
import itertools
from zlib import crc32

from ..models.baseline_constants import MODEL_PARAMS, ACCURACY_KEY
from ..core import Tangle, Transaction, Node, MaliciousNode, PoisonType
from ..core.tip_selection import TipSelector
from .lab_transaction_store import LabTransactionStore


class Lab:
    def __init__(self, tip_selector_factory, config, model_config, node_config, poisoning_config, tx_store=None):
        self.tip_selector_factory = tip_selector_factory
        self.config = config
        self.model_config = model_config
        self.node_config = node_config
        self.poisoning_config = poisoning_config
        self.tx_store = tx_store if tx_store is not None else LabTransactionStore(self.config.tangle_dir, self.config.src_tangle_dir)

        # Set the random seed if provided (affects client sampling, and batching)
        random.seed(1 + config.seed)
        np.random.seed(12 + config.seed)

    @staticmethod
    def create_client_model(seed, model_config):
        model_path = '.%s.%s' % (model_config.dataset, model_config.model)
        mod = importlib.import_module(model_path, package='tangle.models')
        ClientModel = getattr(mod, 'ClientModel')

        # Create 2 models
        model_params = MODEL_PARAMS['%s.%s' % (model_config.dataset, model_config.model)]
        if model_config.lr != -1:
            model_params_list = list(model_params)
            model_params_list[0] = model_config.lr
            model_params = tuple(model_params_list)

        model = ClientModel(seed, *model_params)
        model.num_epochs = model_config.num_epochs
        model.batch_size = model_config.batch_size
        model.num_batches = model_config.num_batches
        return model

    def create_genesis(self):
        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        client_model = self.create_client_model(self.config.seed, self.model_config)

        genesis = Transaction([])
        genesis.add_metadata('time', 0)
        self.tx_store.save(genesis, client_model.get_params())

        return genesis

    def create_node_transaction(self, tangle, round, client_id, cluster_id, train_data, eval_data, seed, model_config, tip_selector, tx_store):

        client_model = Lab.create_client_model(seed, model_config)

        # Choose which nodes are malicious based on a hash, not based on a random variable
        # to have it consistent over the entire experiment run
        # https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
        use_poisoning_node = \
            self.poisoning_config.poison_type != PoisonType.Disabled and \
            self.poisoning_config.poison_from <= round and \
            (float(crc32(client_id.encode('utf-8')) & 0xffffffff) / 2**32) < self.poisoning_config.poison_fraction

        if use_poisoning_node:
            ts = TipSelector(tangle, particle_settings=self.tip_selector_factory.particle_settings) \
                if self.poisoning_config.use_random_ts else tip_selector
            print(f'client {client_id} is is poisoned {"and uses random ts" if self.poisoning_config.use_random_ts else ""}')
            node = MaliciousNode(tangle, tx_store, ts, client_id, cluster_id, train_data, eval_data, client_model, self.poisoning_config.poison_type, config=self.node_config)
        else:
            node = Node(tangle, tx_store, tip_selector, client_id, cluster_id, train_data, eval_data, client_model, config=self.node_config)

        tx, tx_weights = node.create_transaction()

        if tx is not None:
            tx.add_metadata('time', round)

        return tx, tx_weights

    def create_node_transactions(self, tangle, round, clients, dataset):
        tip_selectors = [self.tip_selector_factory.create(tangle) for _ in range(len(clients))]

        result = [self.create_node_transaction(tangle, round, client_id, cluster_id, dataset.train_data[client_id], dataset.test_data[client_id], self.config.seed, self.model_config, tip_selector, self.tx_store)
                  for ((client_id, cluster_id), tip_selector) in zip(clients, tip_selectors)]

        for tx, tx_weights in result:
            if tx is not None:
                self.tx_store.save(tx, tx_weights)

        return [tx for tx, _ in result]

    def create_malicious_transaction(self):
        pass

    def train(self, num_nodes, start_from_round, num_rounds, eval_every, eval_on_fraction, dataset):
        if num_rounds == -1:
            rounds_iter = itertools.count(start_from_round)
        else:
            rounds_iter = range(start_from_round, num_rounds)

        if start_from_round > 0:
            tangle_name = int(start_from_round)-1
            print('Loading previous tangle from round %s' % tangle_name)
            tangle = self.tx_store.load_tangle(tangle_name)

        for round in rounds_iter:
            begin = time.time()
            print('Started training for round %s' % round)
            sys.stdout.flush()

            if round == 0:
                genesis = self.create_genesis()
                tangle = Tangle({genesis.id: genesis}, genesis.id)
            else:
                clients = dataset.select_clients(round, num_nodes)
                print(f"Clients this round: {clients}")
                for tx in self.create_node_transactions(tangle, round, clients, dataset):
                    if tx is not None:
                        tangle.add_transaction(tx)

            print(f'This round took: {time.time() - begin}s')
            sys.stdout.flush()

            self.tx_store.save_tangle(tangle, round)

            if eval_every != -1 and round % eval_every == 0:
                self.print_validation_results(self.validate(round, dataset, eval_on_fraction), round)

    def test_single(self, tangle, client_id, cluster_id, train_data, eval_data, seed, set_to_use, tip_selector):
        import tensorflow as tf

        random.seed(1 + seed)
        np.random.seed(12 + seed)
        tf.compat.v1.set_random_seed(123 + seed)

        client_model = self.create_client_model(seed, self.model_config)
        node = Node(tangle, self.tx_store, tip_selector, client_id, cluster_id, train_data, eval_data, client_model, config=self.node_config)

        reference_txs, reference = node.obtain_reference_params()
        metrics = node.test(reference, set_to_use)
        #if 'clusterId' in tangle.transactions[reference_txs[0]].metadata.keys():
        #    tx_cluster = tangle.transactions[reference_txs[0]].metadata['clusterId']
        #else:
        #    tx_cluster = 'None'
        #if cluster_id != tx_cluster:
        #    with open(os.path.join(os.path.dirname(self.config.tangle_dir), 'validation_nodes.txt'), 'a') as f:
        #        f.write(f'{client_id}({cluster_id}): {reference_txs}({tx_cluster}) (acc: {metrics["accuracy"]:.3f}, loss: {metrics["loss"]:.3f})\n')

        # How many unique poisoned transactions have found their way into the consensus
        # through direct or indirect approvals?

        approved_poisoned_transactions_cache = {}

        def compute_approved_poisoned_transactions(transaction):
            if transaction not in approved_poisoned_transactions_cache:
                tx = tangle.transactions[transaction]
                result = set([transaction]) if 'poisoned' in tx.metadata and tx.metadata['poisoned'] else set([])
                result = result.union(*[compute_approved_poisoned_transactions(parent) for parent in tangle.transactions[transaction].parents])
                approved_poisoned_transactions_cache[transaction] = result

            return approved_poisoned_transactions_cache[transaction]

        approved_poisoned_transactions = set(*[compute_approved_poisoned_transactions(tx) for tx in reference_txs])
        metrics['num_approved_poisoned_transactions'] = len(approved_poisoned_transactions)

        return metrics

    def validate_nodes(self, tangle, clients, dataset):
        tip_selector = self.tip_selector_factory.create(tangle)
        return [self.test_single(tangle, client_id, cluster_id, dataset.train_data[client_id], dataset.test_data[client_id], random.randint(0, 4294967295), 'test', tip_selector) for client_id, cluster_id in clients]

    def validate(self, round, dataset, client_fraction=0.1):
        print('Validate for round %s' % round)
        #import os
        #with open(os.path.join(os.path.dirname(self.config.tangle_dir), 'validation_nodes.txt'), 'a') as f:
        #    f.write('\nValidate for round %s\n' % round)
        tangle = self.tx_store.load_tangle(round)
        if dataset.clients[0][1] is None:
            # No clusters used
            client_indices = np.random.choice(range(len(dataset.clients)),
                                              min(int(len(dataset.clients) * client_fraction), len(dataset.clients)),
                                              replace=False)
        else:
            # validate fairly across all clusters
            client_indices = []
            clusters = np.array(list(map(lambda x: x[1], dataset.clients)))
            unique_clusters = set(clusters)
            num = max(min(int(len(dataset.clients) * client_fraction), len(dataset.clients)), 1)
            div = len(unique_clusters)
            clients_per_cluster = [num // div + (1 if x < num % div else 0)  for x in range(div)]
            for cluster_id in unique_clusters:
                cluster_client_ids = np.where(clusters == cluster_id)[0]
                client_indices.extend(np.random.choice(cluster_client_ids, clients_per_cluster[cluster_id], replace=False))
        validation_clients = [dataset.clients[i] for i in client_indices]
        return self.validate_nodes(tangle, validation_clients, dataset)

    def print_validation_results(self, results, rnd):
        avg_acc = np.average([r[ACCURACY_KEY] for r in results])
        avg_loss = np.average([r['loss'] for r in results])

        avg_message = 'Average %s: %s\nAverage loss: %s' % (ACCURACY_KEY, avg_acc, avg_loss)
        print(avg_message)

        import csv
        import os
        with open(os.path.join(os.path.dirname(self.config.tangle_dir), 'acc_and_loss.csv'), 'a', newline='') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow([rnd, avg_acc, avg_loss])

        write_header = False
        if not os.path.exists(os.path.join(os.path.dirname(self.config.tangle_dir), 'acc_and_loss_all.csv')):
            write_header = True

        with open(os.path.join(os.path.dirname(self.config.tangle_dir), 'acc_and_loss_all.csv'), 'a', newline='') as f:
            for r in results:
                r['round'] = rnd

                r['conf_matrix'] = r['conf_matrix'].tolist()

                w = csv.DictWriter(f, r.keys())
                if write_header:
                    w.writeheader()
                    write_header = False

                w.writerow(r)

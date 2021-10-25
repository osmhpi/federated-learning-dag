import random
import ray

from ..lab import Lab

from . import RayTransactionStore

class RayLab(Lab):
    def __init__(self, tip_selector_factory, config, model_config, node_config, poisoning_config):
        super().__init__(tip_selector_factory, config, model_config, node_config, poisoning_config, tx_store=RayTransactionStore(config.tangle_dir, config.src_tangle_dir))

    def create_genesis(self):
        @ray.remote
        def _create_genesis(self):
            return super().create_genesis()

        genesis = ray.get(_create_genesis.remote(self))

        # Cache it
        self.tx_store.load_transaction_weights(genesis.id)

        return genesis

    @ray.remote
    def create_node_transaction(self, tangle, round, client_id, cluster_id, train_data, eval_data, seed, tip_selector):
        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        train_data = { 'x': ray.get(train_data['x']), 'y': ray.get(train_data['y']) }
        eval_data = { 'x': ray.get(eval_data['x']), 'y': ray.get(eval_data['y']) }

        return super().create_node_transaction(tangle, round, client_id, cluster_id, train_data, eval_data, seed, self.model_config, tip_selector, self.tx_store)

    def create_node_transactions(self, tangle, round, clients, dataset):
        tip_selectors = [self.tip_selector_factory.create(tangle, dataset, client_id, self.tx_store) for (client_id, _) in clients]

        futures = [self.create_node_transaction.remote(self, tangle, round, client_id, cluster_id, dataset.remote_train_data[client_id], dataset.remote_test_data[client_id], random.randint(0, 4294967295), tip_selector)
                   for ((client_id, cluster_id), tip_selector) in zip(clients, tip_selectors)]

        result = ray.get(futures)

        for tx, tx_weights in result:
            if tx is not None:
                self.tx_store.save(tx, tx_weights)

        return [tx for tx, _ in result]

    @ray.remote
    def test_single(self, tangle, client_id, cluster_id, train_data, eval_data, seed, set_to_use, tip_selector):
        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        train_data = { 'x': ray.get(train_data['x']), 'y': ray.get(train_data['y']) }
        eval_data = { 'x': ray.get(eval_data['x']), 'y': ray.get(eval_data['y']) }

        return super().test_single(tangle, client_id, cluster_id, train_data, eval_data, seed, set_to_use, tip_selector)

    def validate_nodes(self, tangle, clients, dataset):
        tip_selectors = [self.tip_selector_factory.create(tangle, dataset, client_id, self.tx_store) for (client_id, _) in clients]

        futures = [self.test_single.remote(self, tangle, client_id, cluster_id, dataset.remote_train_data[client_id], dataset.remote_test_data[client_id], random.randint(0, 4294967295), 'test', tip_selector)
                   for ((client_id, cluster_id), tip_selector) in zip(clients, tip_selectors)]

        return ray.get(futures)

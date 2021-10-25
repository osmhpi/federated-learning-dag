import random
import ray

from ..core.tip_selection.accuracy_tip_selector import AccuracyTipSelectorSettings
from ..core.tip_selection.lazy_accuracy_tip_selector import LazyAccuracyTipSelector
from ..lab import TipSelectorFactory, Lab
from ..models.baseline_constants import ACCURACY_KEY

from .ray_accuracy_tip_selector import RayAccuracyTipSelector

class RayTipSelectorFactory(TipSelectorFactory):
    def __init__(self, config):
        super().__init__(config)
        self.accuracy_cache = {}

    def create(self, tangle, dataset, client_id, tx_store):
        tip_selection_settings = {}
        tip_selection_settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] = self.config.acc_tip_selection_strategy
        tip_selection_settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS] = self.config.acc_cumulate_ratings
        tip_selection_settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] = self.config.acc_ratings_to_weights
        tip_selection_settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] = self.config.acc_select_from_weights
        tip_selection_settings[AccuracyTipSelectorSettings.ALPHA] = self.config.acc_alpha

        if self.config.tip_selector == 'accuracy':
            rayAccuracyTipSelector = RayAccuracyTipSelector(tangle, tip_selection_settings, self.particle_settings)

            if self.config.acc_tip_selection_strategy == 'GLOBAL' and not self.config.acc_cumulate_ratings:
                txs_to_eval = rayAccuracyTipSelector.tips
            else:
                txs_to_eval = tangle.transactions.keys()
            futures = [self.compute_accuracy_ratings.remote(self, client_id, tx_id, random.randint(0, 4294967295), dataset.model_config, dataset.remote_train_data[client_id], tx_store) for tx_id in txs_to_eval]
            currents = ray.get(futures)
            rayAccuracyTipSelector.add_precomputed_ratings({tx_id: r for tx_id, r, _ in currents})

            for tx_id, accuracy, node_id in currents:
                if node_id is not None:
                    if node_id not in self.accuracy_cache:
                        self.accuracy_cache[node_id] = {}
                    self.accuracy_cache[node_id][tx_id] = accuracy

            return rayAccuracyTipSelector

        elif self.config.tip_selector == 'lazy_accuracy':
            # To be consistent with self.config.tip_selector == 'accuracy', draw a random number.
            # If we do not draw a number, in future execution the generated random numbers and thereby train/test data and models will differ
            _ = random.randint(0, 4294967295)
            # Use "normal" LazyAccuracyTipSelector, as there is no need for a ray version of it
            return LazyAccuracyTipSelector(tangle, tip_selection_settings, self.particle_settings)

        return super().create(tangle)

    @ray.remote
    def compute_accuracy_ratings(self, node_id, tx_id, seed, model_config, data, tx_store):
        if node_id in self.accuracy_cache:
            if tx_id in self.accuracy_cache[node_id]:
                return tx_id, self.accuracy_cache[node_id][tx_id], None

        import tensorflow as tf

        # Suppress tf warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        node_model = Lab.create_client_model(seed, model_config)
        node_model.set_params(tx_store.load_transaction_weights(tx_id))

        data = { 'x': ray.get(data['x']), 'y': ray.get(data['y']) }
        accuracy = node_model.test(data)[ACCURACY_KEY]

        return tx_id, accuracy, node_id


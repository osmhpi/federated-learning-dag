import ray
import numpy as np

from ..lab import LabTransactionStore

class RayTransactionStore(LabTransactionStore):
    def __init__(self, tangle_path, src_tangle_path = None):
        super().__init__(tangle_path, src_tangle_path)
        self.tx_cache = {}

    def load_tangle(self, tangle_name):
        tangle = super().load_tangle(tangle_name)

        for id, _ in tangle.transactions.items():
            self.load_transaction_weights(id)

        return tangle

    def load_transaction_weights(self, tx_id):
        if tx_id in self.tx_cache:
            return ray.get(self.tx_cache[tx_id])

        weights = super().load_transaction_weights(tx_id)
        self.tx_cache[tx_id] = ray.put(weights)
        return weights

    def save(self, tx, tx_weights):
        super().save(tx, tx_weights)
        self.tx_cache[tx.id] = ray.put(np.asarray(tx_weights))

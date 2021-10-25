import os
import io
import json
import hashlib
import numpy as np

from ..core import TransactionStore, Transaction, Tangle

class LabTransactionStore(TransactionStore):
    """
    A LabTransactionStore allows to save and load tangles and transactions.
    """

    def __init__(self, dest_tangle_path, src_tangle_path = None):
        """
        Parameters:
        -----------
            `dest_tangle_path`: str
                The path used to store tangles and transactions to.
            `src_tangle_path`: str
                The path used to load pre-existing tangles and transactions from (default: dest_tangle_path)
        """
        self.dest_tangle_path = dest_tangle_path
        self.dest_tx_path = os.path.join(dest_tangle_path, 'transactions')

        if src_tangle_path is not None:
            self.src_tangle_path = src_tangle_path
            self.src_tx_path = os.path.join(src_tangle_path, 'transactions')
        else:
            self.src_tangle_path = self.dest_tangle_path
            self.src_tx_path = self.dest_tx_path

    def load_transaction_weights(self, tx_id):
        # If the transaction belongs to the pre-existing tangle, load it from there,
        # otherwise we generated it and we have to load it from the destination path
        if os.path.exists(f'{self.src_tx_path}/{tx_id}.npy'):
            return np.load(f'{self.src_tx_path}/{tx_id}.npy', allow_pickle=True)
        else:
            return np.load(f'{self.dest_tx_path}/{tx_id}.npy', allow_pickle=True)

    def compute_transaction_id(self, tx_weights):
        tmpfile = io.BytesIO()
        self._save(tx_weights, tmpfile)
        tmpfile.seek(0)
        return self.hash_file(tmpfile)

    def save(self, tx, tx_weights):
        # print(f"saving transaction weights{len(tx_weights)}")
        tx.id = self.compute_transaction_id(tx_weights)

        assert tx_weights, tx.id is not None

        os.makedirs(self.dest_tx_path, exist_ok=True)

        with open(f'{self.dest_tx_path}/{tx.id}.npy', 'wb') as tx_file:
            self._save(tx_weights, tx_file)

    def _save(self, tx_weights, file):
        np.save(file, tx_weights, allow_pickle=True)

    @staticmethod
    def hash_file(f):
        BUF_SIZE = 65536
        sha1 = hashlib.sha1()
        while True:
          data = f.read(BUF_SIZE)
          if not data:
              break
          sha1.update(data)

        return sha1.hexdigest()

    def save_tangle(self, tangle, tangle_name):
        os.makedirs(self.dest_tangle_path, exist_ok=True)

        n = [{'id': t.id,
              'parents': list(t.parents),
              'metadata': t.metadata } for _, t in tangle.transactions.items()]

        with open(f'{self.dest_tangle_path}/tangle_{tangle_name}.json', 'w') as outfile:
            json.dump({'nodes': n, 'genesis': tangle.genesis}, outfile)

        tangle.name = tangle_name

    def load_tangle(self, tangle_name):
        # If the tangle belongs to the pre-existing tangle(s), load it from there,
        # otherwise we generated it and we have to load it from the destination path
        if os.path.exists(f'{self.src_tangle_path}/tangle_{tangle_name}.json'):
            with open(f'{self.src_tangle_path}/tangle_{tangle_name}.json', 'r') as tanglefile:
                t = json.load(tanglefile)
        else:
            with open(f'{self.dest_tangle_path}/tangle_{tangle_name}.json', 'r') as tanglefile:
                t = json.load(tanglefile)

        transactions = {n['id']: Transaction(
                                    set(n['parents']),
                                    n['id'],
                                    n['metadata']
                                 ) for n in t['nodes']}
        tangle = Tangle(transactions, t['genesis'])
        tangle.name = tangle_name
        return tangle

from abc import ABC, abstractmethod

class TransactionStore(ABC):
    @abstractmethod
    def load_transaction_weights(self, tx_id):
        pass

    @abstractmethod
    def compute_transaction_id(self, tx):
        pass

    @abstractmethod
    def save(self, tx, tx_weights):
        pass

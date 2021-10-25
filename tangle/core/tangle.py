import itertools

class Tangle:
    def __init__(self, transactions, genesis):
        self.transactions = transactions
        self.genesis = genesis
        self.tips = self.find_tips(transactions)
        self.depth_cache = self.calculate_depth()

    def calculate_depth(self):
        """
        Returns dict with {depth: [transaction_ids]}

        Depth is defined as the length of the longest directed path from a tip to the transaction.
        """

        # first calculate depth for each transaction

        depth_per_transaction = {}
        depth = 0
        current_transactions = [tx for _, tx in self.transactions.items() if tx.id in self.tips]

        for t in current_transactions:
            depth_per_transaction[t.id] = depth
        
        depth += 1
        parents = set(itertools.chain(*[tx.parents for tx in current_transactions]))

        while len(parents) > 0:
            for parent in parents:
                depth_per_transaction[parent] = depth

            depth += 1
            current_transactions = [tx for _, tx in self.transactions.items() if tx.id in parents]
            parents = set(itertools.chain(*[tx.parents for tx in current_transactions]))

        # build desired dict structure
        
        transactions_per_depth = {}

        for d in range(depth):
            transactions_per_depth[d] = [tx for tx, tx_depth in depth_per_transaction.items() if tx_depth == d]
        
        return transactions_per_depth

    def find_tips(self, transactions):
        potential_tips = set(transactions.keys())
        for _, tx in transactions.items():
            for parent_tx in tx.parents:
                potential_tips.discard(parent_tx)
        return potential_tips

    def add_transaction(self, tip):
        self.transactions[tip.id] = tip
        for parent_tx in tip.parents:
            self.tips.discard(parent_tx)
        self.tips.add(tip.id)

    def get_transaction_ids_of_depth_interval(self, depth_start, depth_end):
        """
        Returns all transaction ids from the tangle, which have a depth between or equal depth_start and depth_end.
        """
        gathered_transaction_ids = []

        for depth in range(depth_start, depth_end + 1):
            if depth in self.depth_cache:
                gathered_transaction_ids.extend(self.depth_cache[depth])
        
        # If no transaction was found inside this interval return genesis
        if len(gathered_transaction_ids) == 0:
            gathered_transaction_ids.append(self.genesis)
        
        return gathered_transaction_ids

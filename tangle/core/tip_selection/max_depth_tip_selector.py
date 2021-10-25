from collections import Counter
import numpy as np

from . import TipSelector

class MaxDepthTipSelector(TipSelector):
    """This tip selector's run time is constant with regards to the depth of the tangle"""

    def __init__(self, tangle, depth):
        entrypoint_hits = []
        rated_transactions = set()

        def recurse_parents(tx, d):
            if tx.id in rated_transactions:
                return

            rated_transactions.add(tx.id)

            if d == 0 or len(tx.parents) == 0:
                # Since we never iterate over a transaction twice,
                # this only counts direct approvals
                entrypoint_hits.append(tx.id)
                return

            for parent in tx.parents:
                recurse_parents(tangle.transactions[parent], d-1)

        for tip in tangle.tips:
            recurse_parents(tangle.transactions[tip], depth)

        c = Counter(entrypoint_hits)
        trunk, _ = c.most_common(1)[0]
        branch = np.random.choice(list(c))

        super().__init__(tangle, trunk, branch, rated_transactions=rated_transactions)

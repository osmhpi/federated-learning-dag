from enum import Enum
import random

import numpy as np

# https://docs.iota.org/docs/node-software/0.1/iri/references/iri-configuration-options
DEFAULT_ALPHA = 0.001

class TipSelectorSettings(Enum):
    USE_PARTICLES = 0
    PARTICLES_DEPTH_START = 1
    PARTICLES_DEPTH_END = 2
    NUM_PARTICLES = 3

class TipSelector:
    def __init__(self, tangle, trunk=None, branch=None, rated_transactions=None, particle_settings={TipSelectorSettings.USE_PARTICLES: False}):
        self.tangle = tangle
        self.ratings = None
        self.particle_settings = particle_settings

        # 'Particles' are starting points for the tip selection walk.
        # The 'trunk' is supposed to reside 'in the center' of the tangle,
        # whereas the 'branch' may lie on the outside.
        self.trunk = trunk if trunk is not None else self.tangle.genesis
        self.branch = branch if branch is not None else self.tangle.genesis
        self.rated_transactions = rated_transactions if rated_transactions is not None else set(
            self.tangle.transactions.keys())

        # Build a map of transactions that directly approve a given transaction
        self.approving_transactions = {x: [] for x in self.rated_transactions}
        for x in self.rated_transactions:
            for unique_parent in self.tangle.transactions[x].parents:
                if unique_parent not in self.rated_transactions:
                    continue
                self.approving_transactions[unique_parent].append(x)

        self.trace = []

    def tx_rating(self, tx, node):
        return self.ratings[tx]

    def tip_selection(self, num_tips, node):
        # https://docs.iota.org/docs/node-software/0.1/iri/concepts/tip-selection
        # The docs say entry_point = latestSolidMilestone - depth.
        tips = []

        if self.particle_settings[TipSelectorSettings.USE_PARTICLES]:
            num_particles = self.particle_settings[TipSelectorSettings.NUM_PARTICLES]
            depth_start = self.particle_settings[TipSelectorSettings.PARTICLES_DEPTH_START]
            depth_end = self.particle_settings[TipSelectorSettings.PARTICLES_DEPTH_END]

            particles = self.tangle.get_transaction_ids_of_depth_interval(depth_start=depth_start, depth_end=depth_end)

            # num_particles cannot be greater than the number of transactions, which could act as particles
            if len(particles) < num_particles:
                num_particles = len(particles)

            # randomly reduce particles to num_particles
            random.shuffle(particles)
            particles = particles[:num_particles]

            for _ in range(num_tips):
                # select particle
                start_tx = self._select_particle(particles, node)
                tips.append(self.walk(start_tx, node, self.approving_transactions))

        else:
            # Start from the 'branch' once
            tips.append(self.walk(self.branch, node, self.approving_transactions))

            for _ in range(num_tips-1):
                # Start walking from the 'trunk' for all remaining tips
                tips.append(self.walk(self.trunk, node, self.approving_transactions))

        return tips
    
    def _select_particle(self, particles, node):
        return random.choice(particles)

    def compute_ratings(self, node):
        rating = {}
        future_set_cache = {}
        for tx in self.rated_transactions:
            rating[tx] = len(TipSelector.future_set(tx, self.approving_transactions, future_set_cache)) + 1

        self.ratings = rating

    def walk(self, tx, node, approving_transactions):
        step = tx
        prev_step = None

        while step is not None:
            approvers = approving_transactions[step]
            prev_step = step
            step = self.next_step(approvers, node)

        # When there are no more steps, this transaction is a tip
        return prev_step

    def next_step(self, approvers, node):

        # If there is no valid approver, this transaction is a tip
        if len(approvers) == 0:
            return None

        if len(approvers) == 1:
            # print("Only one approver")
            return approvers[0]

        approvers_ratings = [self.tx_rating(a, node) for a in approvers]
        weights = self.ratings_to_weight(approvers_ratings)
        approver = self.weighted_choice(approvers, weights)


        trace_of_this_step = zip(approvers, [self.tangle.transactions[approver_id].metadata['issuer'] for approver_id in approvers], approvers_ratings, weights)
        self.trace.append((list(trace_of_this_step), approver, self.tangle.transactions[approver].metadata['issuer']))
        # print("Approvers: ")
        # print([self.tangle.transactions[approver_id].metadata['issuer'] for approver_id in approvers])
        # print(approvers_ratings)
        # print(weights)
        # print(f"{self.tangle.transactions[approver].metadata['issuer'] }({approver})")
        # Skip validation.
        # At least a validation of some PoW is necessary in a real-world implementation.

        return approver

        # if approver is not None:
        #     tail = validator.findTail(approver)
        #
        #     # If the selected approver is invalid, step back and try again
        #     if validator.isInvalid(tail):
        #         approvers = approvers.remove(approver)
        #
        #         return self.next_step(ratings, approvers)
        #
        #     return tail
        #
        # return None

    def normalize_ratings(self, ratings, dynamic=True):
        # with dynamic = True, the ratings are linearly mapped to a scale between -1 and 0
        highest_rating = max(ratings)
        lowest_rating = min(ratings)
        rating_spread = highest_rating - lowest_rating

        normalized_ratings = [rating - highest_rating for rating in ratings]

        if dynamic and rating_spread != 0:
            normalized_ratings = [rating / rating_spread for rating in normalized_ratings]

        return normalized_ratings

    def ratings_to_weight(self, ratings, alpha=DEFAULT_ALPHA, dynamic=True):
        normalized_ratings = self.normalize_ratings(ratings, dynamic)
        return [np.exp(r * alpha) for r in normalized_ratings]

    def weighted_choice(self, approvers, weights):

        rn = random.uniform(0, sum(weights))
        for i in range(len(approvers)):
            rn -= weights[i]
            if rn <= 0:
                return approvers[i]
        return approvers[-1]

    @staticmethod
    def future_set(tx, approving_transactions, future_set_cache):
        def recurse_future_set(t):
            if t not in future_set_cache:
                direct_approvals = set(approving_transactions[t])
                future_set_cache[t] = direct_approvals.union(*[recurse_future_set(x) for x in direct_approvals])

            return future_set_cache[t]

        return recurse_future_set(tx)
    #
    # ratings_to_probability is not used currently.
    # @staticmethod
    # def ratings_to_probability(ratings):
    #     # Calculating a probability according to the IOTA randomness blog
    #     # https://blog.iota.org/alpha-d176d7601f1c
    #     b = sum(map(lambda r: np.exp(DEFAULT_ALPHA * r), ratings))
    #     return [np.exp(r * DEFAULT_ALPHA) / b for r in ratings]

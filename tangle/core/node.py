from .transaction import Transaction

class NodeConfiguration:
    num_tips: int
    sample_size: int
    reference_avg_top: int
    publish_if_better_than: str

    def __init__(self, num_tips=2, sample_size=2, reference_avg_top=1, publish_if_better_than='REFERENCE'):
        self.num_tips = num_tips
        self.sample_size = sample_size
        self.reference_avg_top = reference_avg_top
        self.publish_if_better_than = publish_if_better_than

class Node:
    def __init__(self, tangle, tx_store, tip_selector, client_id, cluster_id, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None, config=NodeConfiguration()):
        self.tangle = tangle
        self.tx_store = tx_store
        self.tip_selector = tip_selector
        self._model = model
        self.id = client_id
        self.cluster_id = cluster_id
        self.train_data = train_data
        self.eval_data = eval_data
        self.config = config

        # Initialize tip selector
        tip_selector.compute_ratings(self)

    @staticmethod
    def average_model_params(*params):
        return sum(params) / len(params)

    def train(self, model_params):
        """Trains on self.model using the client's train_data.

        Args:
            model_params: params that are used as basis for the training

        Returns:
            model params of the new model after training
        """
        self.model.set_params(model_params)

        data = self.train_data
        self.model.train(data)
        # update = self.model.train(data)
        # num_train_samples = len(data['y'])
        # return num_train_samples, update
        return self.model.get_params()

    def test(self, model_params, set_to_use='test', only_test_on_first_1000=False):
        """Tests self.model on self.test_data.

        Args:
            set_to_use: Set to test on. Should be in ['train', 'test'].

        Returns:
            dict of metrics returned by the model.
        """
        self.model.set_params(model_params)

        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        if(only_test_on_first_1000):
            data = {'x': data['x'][:1000], 'y': data['y'][:1000]}
        # begin = time.time()
        metrics = self.model.test(data)
        # print(f'Testing took: {time.time()-begin}')
        return metrics

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Returns:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Returns:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Returns:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def choose_tips(self, num_tips=2, sample_size=2):
        if len(self.tangle.transactions) < num_tips:
            return [self.tangle.transactions[self.tangle.genesis] for i in range(2)]

        tips = self.tip_selector.tip_selection(sample_size, self)

        no_dups = set(tips)
        if len(no_dups) >= num_tips:
            tips = no_dups

        tip_txs = [self.tangle.transactions[tip] for tip in tips]

        # Find best tips
        if num_tips < sample_size:
            # Choose tips with lowest test loss
            tip_losses = []
            loss_cache = {}
            for tip in tip_txs:
                if tip.id in loss_cache.keys():
                    tip_losses.append((tip, loss_cache[tip.id]))
                else:
                    loss = self.test(self.tx_store.load_transaction_weights(tip.id), 'test')['loss']
                    tip_losses.append((tip, loss))
                    loss_cache[tip.id] = loss
            best_tips = sorted(tip_losses, key=lambda tup: tup[1], reverse=False)[:num_tips]
            tip_txs = [tup[0] for tup in best_tips]

        return tip_txs

    def compute_confidence(self, approved_transactions_cache={}):
        num_sampling_rounds = 5

        transaction_confidence = {x: 0 for x in self.tangle.transactions}

        def approved_transactions(transaction):
            if transaction not in approved_transactions_cache:
                result = set([transaction]).union(*[approved_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
                approved_transactions_cache[transaction] = result

            return approved_transactions_cache[transaction]

        for i in range(num_sampling_rounds):
            tips = self.choose_tips()
            for tip in tips:
                for tx in approved_transactions(tip.id):
                    transaction_confidence[tx] += 1

        return {tx: float(transaction_confidence[tx]) / (num_sampling_rounds * 2) for tx in self.tangle.transactions}

    def compute_cumulative_score(self, transactions, approved_transactions_cache={}):
        def compute_approved_transactions(transaction):
            if transaction not in approved_transactions_cache:
                result = set([transaction]).union(*[compute_approved_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
                approved_transactions_cache[transaction] = result

            return approved_transactions_cache[transaction]

        return {tx: len(compute_approved_transactions(tx)) for tx in transactions}

    def obtain_reference_params(self, avg_top=1):
        # Establish the 'current best'/'reference' weights from the tangle

        approved_transactions_cache = {}

        # 1. Perform tip selection n times, establish confidence for each transaction
        # (i.e. which transactions were already approved by most of the current tips?)
        transaction_confidence = self.compute_confidence(approved_transactions_cache=approved_transactions_cache)

        # 2. Compute cumulative score for transactions
        # (i.e. how many other transactions does a given transaction indirectly approve?)
        keys = [x for x in self.tangle.transactions]
        scores = self.compute_cumulative_score(keys, approved_transactions_cache=approved_transactions_cache)

        # 3. For the top avg_top transactions, compute the average
        best = sorted(
            {tx: scores[tx] * transaction_confidence[tx] for tx in keys}.items(),
            key=lambda kv: kv[1], reverse=True
        )[:avg_top]
        reference_txs = [elem[0] for elem in best]
        reference_params = Node.average_model_params(*[self.tx_store.load_transaction_weights(elem) for elem in reference_txs])
        return reference_txs, reference_params

    def create_transaction(self):

        # Obtain number of tips from the tangle
        tips = self.choose_tips(num_tips=self.config.num_tips, sample_size=self.config.sample_size)

        # Perform averaging

        # How averaging is done exactly (e.g. weighted, using which weights) is left to the
        # network participants. It is not reproducible or verifiable by other nodes because
        # only the resulting weights are published.
        # Once a node has published its training results, it thus can't be sure if
        # and by what weight its delta is being incorporated into approving transactions.
        # However, assuming most nodes are well-behaved, they will make sure that eventually
        # those weights will prevail that incorporate as many partial results as possible
        # in order to prevent over-fitting.

        # Here: simple unweighted average
        tx_weights = [self.tx_store.load_transaction_weights(tip.id) for tip in tips]

        averaged_params = Node.average_model_params(*tx_weights)
        averaged_model_metrics = self.test(averaged_params, 'test')

        trained_params = self.train(averaged_params)
        trained_model_metrics = self.test(trained_params, 'test')

        transaction = None

        assert self.config.publish_if_better_than in ['PARENTS', 'REFERENCE']
        if(self.config.publish_if_better_than == 'REFERENCE'):
            print("publish if better than reference")
            # Compute reference metrics
            reference_txs, reference_params = self.obtain_reference_params(avg_top=self.config.reference_avg_top)
            reference_metrics = self.test(reference_params, 'test')
            if trained_model_metrics['loss'] < reference_metrics['loss']:
                print("i'll publish!")
                transaction = Transaction(parents=set([tip.id for tip in tips]))
                transaction.add_metadata('reference_tx', reference_txs[0])
                transaction.add_metadata('reference_tx_loss', float(reference_metrics['loss']))
                transaction.add_metadata('reference_tx_accuracy', reference_metrics['accuracy'])
        else:
            print("publish if better than parents")
            if trained_model_metrics['loss'] < averaged_model_metrics['loss']:
                print("i'll publish!")
                transaction = Transaction(parents=set([tip.id for tip in tips]))

        if transaction is not None:
            transaction.add_metadata('issuer', self.id)
            transaction.add_metadata('issuer_data_size', len(self.train_data))
            transaction.add_metadata('clusterId', self.cluster_id)
            transaction.add_metadata('loss', float(trained_model_metrics['loss']))
            transaction.add_metadata('accuracy', trained_model_metrics['accuracy'])
            transaction.add_metadata('averaged_loss', float(averaged_model_metrics['loss']))
            transaction.add_metadata('averaged_accuracy', averaged_model_metrics['accuracy'])
            transaction.add_metadata('trace', self.tip_selector.trace)

        return transaction, trained_params

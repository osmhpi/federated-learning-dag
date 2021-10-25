import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from skimage.future.graph import cut_normalized

from tangle.lab.lab_transaction_store import LabTransactionStore


# ------------------- Helper functions --------------------
def get_cluster_id(tx_name, tangle):
    """ Returns cluster ID of given transaction.

    Args:
        tx_name: the transaction name.
        tangle: the tangle.

    Returns:
        int: The cluster ID.
    """
    if 'clusterId' not in tangle.transactions[tx_name].metadata:
        return None
    return tangle.transactions[tx_name].metadata['clusterId']


def future_set(tx, approving_transactions, future_set_cache={}):
    """ Helper function to compute future set of approving transactions for a given transaction.

    Args:
        tx: the transaction name.
        approving_transactions: a dict containing all directly approving txs for each tx.
        future_set_cache: cache to reduce time for computation.

    Returns:
        set: A set of all directly and indirectly approving transactions.
    """
    def recurse_future_set(t):
        if t not in future_set_cache:
            direct_approvals = set(approving_transactions[t])
            future_set_cache[t] = direct_approvals.union(*[recurse_future_set(x) for x in direct_approvals])

        return future_set_cache[t]

    return recurse_future_set(tx)


def create_networkx_from_tangle(tangle):
    """ Converts tangle data structure to networkx directed graph.

    Args:
        tangle: the tangle.

    Returns:
        DiGraph: networkx directed graph representing given tangle
    """
    G = nx.DiGraph()
    for x, tx in tangle.transactions.items():
        # add node and attribute
        G.add_node(x, cluster=get_cluster_id(x, tangle))
        G.add_edges_from(list(zip([x]*len(tx.parents), tx.parents)))
    print('created graph with {:d} nodes and {:d} edges'.format(G.number_of_nodes(), G.size()))
    return G

# -----------------------------------------


def compute_within_cluster_approval_fraction(tangle, num_cluster=4):
    """ Computes the fraction of direct and indirect approvals within clusters
    (where parent and child cluster ID match).

    Args:
        tangle: the tangle.
        num_cluster: the number of clusters in the data.

    Returns:
        dict: cluster -> (absolute number of direct and indirect within-cluster approvals,
                          absolute number of direct and indirect cluster approvals).
        dict: cluster -> fraction of direct and indirect within-cluster approvals.
    """
    print('Computing within-cluster approval fraction (direct and indirect)...')
    approving_transactions = {x: [] for x in tangle.transactions}
    for x, tx in tangle.transactions.items():
        for unique_parent in tx.parents:
            approving_transactions[unique_parent].append(x)

    future_set_cache = {}
    future_sets = {}
    for tx in tangle.transactions:
        future_sets[tx] = future_set(tx, approving_transactions, future_set_cache)

    cluster_absolutes = {}  # Absolute number of within-cluster approvals
    cluster_ratings = {}    # Relative number of within-cluster approvals
    for i in range(num_cluster):
        cluster_absolutes[i] = (0, 0)  # within_cluster_direct_approvals, total_cluster_approvals,

    for tx, future in future_sets.items():
        cluster_id = get_cluster_id(tx, tangle)
        for transaction in future:
            tx_cluster_id = get_cluster_id(transaction, tangle)
            if tx_cluster_id == cluster_id:
                add_tuple = (1, 1)
            else:
                add_tuple = (0, 1)
            approvals = cluster_absolutes[tx_cluster_id][0] + add_tuple[0]
            totals = cluster_absolutes[tx_cluster_id][1] + add_tuple[1]
            cluster_absolutes[tx_cluster_id] = (approvals, totals)

    for i in range(num_cluster):
        cluster_ratings[i] = cluster_absolutes[i][0] / cluster_absolutes[i][1]
        print('Cluster {:d}: {:.1f}%'.format(i, 100 * cluster_ratings[i]))

    return cluster_absolutes, cluster_ratings


def compute_within_cluster_direct_approval_fraction(tangle, num_cluster=4):
    """ Computes the fraction of direct approvals in the tangle within clusters
    (where parent and child cluster ID match.)

    Args:
        tangle: the tangle.
        num_cluster: the number of clusters in the data.

    Returns:
        dict: cluster -> (absolute number of direct within-cluster approvals,
                          absolute number of direct cluster approvals).
        dict: cluster -> fraction of direct within-cluster approvals.
    """
    print('Computing within-cluster approval fraction (direct only)...')
    cluster_absolutes = {}  # Absolute number of within-cluster approvals
    cluster_ratings = {}  # Relative number of within-cluster approvals
    for i in range(num_cluster):
        cluster_absolutes[i] = (0, 0)     # within_cluster_direct_approvals, total_cluster_approvals,
    for x, tx in tangle.transactions.items():
        if 'clusterId' not in tx.metadata:
            continue
        cluster_id = tx.metadata['clusterId']
        within_cluster_direct_approvals = 0
        for unique_parent in tx.parents:
            parent_cluster_id = get_cluster_id(unique_parent, tangle)
            if cluster_id == parent_cluster_id:
                within_cluster_direct_approvals = within_cluster_direct_approvals + 1
        approvals = cluster_absolutes[cluster_id][0] + within_cluster_direct_approvals
        totals = cluster_absolutes[cluster_id][1] + len(tx.parents)
        cluster_absolutes[cluster_id] = (approvals, totals)

    for i in range(num_cluster):
        cluster_ratings[i] = cluster_absolutes[i][0] / cluster_absolutes[i][1]
        print('Cluster {:d}: {:.1f}%'.format(i, 100 * cluster_ratings[i]))

    return cluster_absolutes, cluster_ratings


def get_within_cluster_subgraphs(tangle, num_cluster=4):
    """ Computes and prints the number of 'weakly connected components' in subgraphs built from all transactions of each
    cluster using existing networkx function number_weakly_connected_components.

    Args:
        tangle: the tangle.
        num_cluster: the number of clusters in the data.

    Returns:
        list: subgraphs of each cluster.
    """
    # Remove all non-within-cluster approvals and check for connectedness
    graphs = []
    for i in range(num_cluster):
        graphs.append(nx.DiGraph())

    for x, tx in tangle.transactions.items():
        # add node and attribute
        if 'clusterId' not in tx.metadata:
            continue
        cluster_id = tx.metadata['clusterId']
        graphs[cluster_id].add_node(x)
        for unique_parent in tx.parents:
            if tx.metadata['clusterId'] == get_cluster_id(unique_parent, tangle):
                graphs[cluster_id].add_edge(x, unique_parent)

    for i in range(num_cluster):
        print('Cluster {}: {} weakly connected components'.format(i, nx.number_weakly_connected_components(graphs[i])))

    return graphs


def draw_greedy_modularity_communities(graph):
    """ Applies greedy_modularity_communities function to tangle graph and draws communities to Pyplot.

    Args:
        graph: a tangle graph.
    """
    communities = nx.algorithms.community.greedy_modularity_communities(graph.to_undirected())
    print('Found {} greedy modularity communities'.format(len(communities)))
    pos = nx.spring_layout(graph)
    cmap = plt.cm.get_cmap('hsv')
    color_list = np.linspace(0, 1, len(communities))
    for i in range(len(communities)):
        nx.draw_networkx_nodes(graph, pos, nodelist=communities[i], node_size=10, node_color=np.reshape(cmap(color_list[i]), (1, -1)))
    nx.draw_networkx_edges(graph, pos, edge_color='black')
    plt.axis('off')
    plt.show()


def get_avg_txs_per_round(tangle):
    tangle = tangle.transactions
    tx_times = list(map(lambda elem: elem.metadata['time'], tangle))
    c = Counter(tx_times)
    avg = np.mean(list(c.values()))
    print(f'Average number of TXs per round: {avg}')


def normalized_cut(graph):
    """ WORK-IN-PROGRESS - computes normalized cut of tangle graph.

    Args:
        graph: a tangle graph.
    """
    labels = cut_normalized(range(10), graph.to_undirected())
    print(labels)


def parse_args():
    parser = argparse.ArgumentParser(description='Graph analysis of tangle results')
    parser.add_argument('--name',
                        help='The name of the experiment. Folder name in ../experiments/<name>. Default: <dataset>-<model>-<exp_number>')
    parser.add_argument('--config',
                        default='0',
                        help='The config ID of the experiment.')
    parser.add_argument('--epoch',
                        help='The tangle epoch to analyse.')
    parser.add_argument('--num-cluster',
                        default=3,
                        type=int,
                        help='The number of clusters in the data.')
    return parser.parse_args()


def main():
    args = parse_args()

    tx_store = LabTransactionStore(f'../experiments/{args.name}/config_{args.config}/tangle_data')
    tangle = tx_store.load_tangle(args.epoch)

    cluster_approvals, cluster_rating = compute_within_cluster_approval_fraction(tangle, num_cluster=args.num_cluster)

    cluster_approvals, cluster_rating = compute_within_cluster_direct_approval_fraction(tangle, num_cluster=args.num_cluster)

    # graph = create_networkx_from_tangle(tangle)
    # normalized_cut(graph)
    # draw_greedy_modularity_communities(graph)
    get_within_cluster_subgraphs(tangle, num_cluster=args.num_cluster)

    get_avg_txs_per_round(tangle)


if __name__ == "__main__":
    main()


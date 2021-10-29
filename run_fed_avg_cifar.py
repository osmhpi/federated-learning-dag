import subprocess

params = {
    'dataset': 'cifar100',   # is expected to be one value to construct default experiment name
    'model': 'cnn',       # is expected to be one value to construct default experiment name
    'num_rounds': 100,
    'eval_every': 101,
    'eval_on_fraction': 0.05,
    'clients_per_round': 10,
    'model_data_dir': './data/cifar-joined_five_clients',
    'batch_size': 10,
    'num_batches': 90,
    'learning_rate': 0.01
}

def main():
    command = 'python -m fedavg ' \
                '-dataset %s ' \
                '-model %s ' \
                '--num-rounds %s ' \
                '--eval-every %s ' \
                '--eval-on-fraction %s ' \
                '--clients-per-round %s ' \
                '--model-data-dir %s ' \
                '--batch-size %s ' \
                '--num-batches %s ' \
                '-lr %s'
    parameters = (
        params['dataset'],
        params['model'],
        params['num_rounds'],
        params['eval_every'],
        params['eval_on_fraction'],
        params['clients_per_round'],
        params['model_data_dir'],
        params['batch_size'],
        params['num_batches'],
        params['learning_rate'])
    command = command % parameters

    print('Training started...')
    with open('fed_avg_output.txt', 'w+') as out_file:
        training = subprocess.Popen(command.split(" "), stdout=out_file)
        training.wait()
    print('Training finished...')


if __name__ == '__main__':
    main()

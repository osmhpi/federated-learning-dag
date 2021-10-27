import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

from tangle.analysis import TangleAnalysator

from sklearn.model_selection import ParameterGrid

#############################################################################
############################# Parameter section #############################
#############################################################################

params = {
    'dataset': ['femnist'],   # is expected to be one value to construct default experiment name
    'model': ['cnn'],      # is expected to be one value to construct default experiment name
    'num_rounds': [200],
    'eval_every': [5],
    'eval_on_fraction': [0.05],
    'clients_per_round': [10],
    'model_data_dir': ['./data/fmnist'],
    'src_tangle_dir': ['./experiments/<insert pretrain experiment name>/config_0/tangle_data'],         # Set to '' to not use --src-tangle-dir parameter
    'start_round': [100],
    'tip_selector': ['lazy_accuracy'],
    'num_tips': [2],
    'sample_size': [2],
    'batch_size': [10],
    'num_batches': [10],
    'publish_if_better_than': ['REFERENCE'], # or parents
    'reference_avg_top': [1],
    'target_accuracy': [1],
    'learning_rate': [0.05],
    'num_epochs': [1],
    'acc_tip_selection_strategy': ['WALK'],
    'acc_cumulate_ratings': ['False'],
    'acc_ratings_to_weights': ['ALPHA'],
    'acc_select_from_weights': ['WEIGHTED_CHOICE'],
    'acc_alpha': [10],
    'use_particles': ['False'],
    'particles_depth_start': [10],
    'particles_depth_end': [20],
    'particles_number': [10],
    'poison_type': ['labelflip'],
    'poison_fraction': [0, 0.2, 0.3],
    'poison_from': [0],
    'poison_use_random_ts': ['False'],
}

##############################################################################
########################## End of Parameter section ##########################
##############################################################################

def main():
    setup_filename = '1_setup.log'
    console_output_filename = '2_training.log'

    # exit_if_repo_not_clean()

    args = parse_args()
    experiment_folder = prepare_exp_folder(args)

    print("[Info]: Experiment results and log data will be stored at %s" % experiment_folder)

    git_hash = get_git_hash()
    run_and_document_experiments(args, experiment_folder, setup_filename, console_output_filename, git_hash)

def exit_if_repo_not_clean():
    proc = subprocess.Popen(['git', 'status', '--porcelain'], stdout=subprocess.PIPE)

    try:
        dirty_files, errs = proc.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, errs = proc.communicate()
        print('[Error]: Could not check git status!: %s' % errs, file=sys.stderr)
        exit(1)

    if dirty_files:
        print('[Error]: You have uncommited changes. Please commit them before continuing. No experiments will be executed.', file=sys.stderr)
        exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Run and document an experiment.')
    parser.add_argument('--name', help='The name of the experiment. Results will be stored under ../experiments/<name>. Default: <dataset>-<model>-<exp_number>')
    parser.add_argument('--overwrite_okay', type=bool, default=False, help='Overwrite existing experiment with same name. Default: False')
    args = parser.parse_args()

    return args

def prepare_exp_folder(args):
    experiments_base = './experiments'
    os.makedirs(experiments_base, exist_ok=True)

    if not args.name:
        default_prefix = "%s-%s" % (params['dataset'][0], params['model'][0])

        # Find other experiments with default names
        all_experiments = next(os.walk(experiments_base))[1]
        default_exps = [exp for exp in all_experiments if re.match("^(%s-\d+)$" % default_prefix, exp)]

        # Find the last experiments with default name and increment id
        if len(default_exps) == 0:
            next_default_exp_id = 0
        else:
            default_exp_ids = [int(exp.split("-")[-1]) for exp in default_exps]
            default_exp_ids.sort()
            next_default_exp_id = default_exp_ids[-1] + 1

        args.name = "%s-%d" % (default_prefix, next_default_exp_id)

    exp_name = args.name

    experiment_folder = experiments_base + '/' + exp_name

     # check, if existing experiment exists
    if (os.path.exists(experiment_folder) and not args.overwrite_okay):
        print('[Error]: Experiment "%s" already exists! To overwrite set --overwrite_okay to True' % exp_name, file=sys.stderr)
        exit(1)

    os.makedirs(experiment_folder, exist_ok=True)

    return experiment_folder

def get_git_hash():
    proc = subprocess.Popen(['git', 'rev-parse', '--verify', 'HEAD'], stdout=subprocess.PIPE)
    try:
        git_hash, errs = proc.communicate(timeout=3)
        git_hash = git_hash.decode("utf-8")
    except subprocess.TimeoutExpired:
        proc.kill()
        _, errs = proc.communicate()
        git_hash = 'Could not get Githash!: %s' % errs

    return git_hash

def run_and_document_experiments(args, experiments_dir, setup_filename, console_output_filename, git_hash):

    shutil.copy(__file__, experiments_dir)

    parameter_grid = ParameterGrid(params)
    print(f'Starting experiments for {len(parameter_grid)} parameter combinations...')
    for idx, p in enumerate(parameter_grid):
        # Create folder for that run
        experiment_folder = experiments_dir + '/config_%s' % idx
        os.makedirs(experiment_folder, exist_ok=True)

        # Prepare execution command
        command = 'python -m tangle.ray ' \
            '-dataset %s ' \
            '-model %s ' \
            '--num-rounds %s ' \
            '--eval-every %s ' \
            '--eval-on-fraction %s ' \
            '--clients-per-round %s ' \
            '--tangle-dir %s ' \
            '--model-data-dir %s ' \
            '--target-accuracy %s ' \
            '--num-tips %s ' \
            '--sample-size %s ' \
            '--batch-size %s ' \
            '--num-batches %s ' \
            '-lr %s ' \
            '--num-epochs %s ' \
            '--publish-if-better-than %s ' \
            '--reference-avg-top %s ' \
            '--tip-selector %s ' \
            '--acc-tip-selection-strategy %s ' \
            '--acc-cumulate-ratings %s ' \
            '--acc-ratings-to-weights %s ' \
            '--acc-select-from-weights %s ' \
            '--acc-alpha %s ' \
            '--use-particles %s ' \
            '--particles-depth-start %s ' \
            '--particles-depth-end %s ' \
            '--particles-number %s ' \
            '--poison-type %s ' \
            '--poison-fraction %s ' \
            '--poison-from %s ' \
            '--poison-use-random-ts %s ' \
            ''
        parameters = (
            p['dataset'],
            p['model'],
            p['num_rounds'],
            p['eval_every'],
            p['eval_on_fraction'],
            p['clients_per_round'],
            experiment_folder + '/tangle_data',
            p['model_data_dir'],
            p['target_accuracy'],
            p['num_tips'],
            p['sample_size'],
            p['batch_size'],
            p['num_batches'],
            p['learning_rate'],
            p['num_epochs'],
            p['publish_if_better_than'],
            p['reference_avg_top'],
            p['tip_selector'],
            p['acc_tip_selection_strategy'],
            p['acc_cumulate_ratings'],
            p['acc_ratings_to_weights'],
            p['acc_select_from_weights'],
            p['acc_alpha'],
            p['use_particles'],
            p['particles_depth_start'],
            p['particles_depth_end'],
            p['particles_number'],
            p['poison_type'],
            p['poison_fraction'],
            p['poison_from'],
            p['poison_use_random_ts'],
        )
        command = command.strip() % parameters

        if len(p['src_tangle_dir']) > 0:
            command = '%s --src-tangle-dir %s' % (command, p['src_tangle_dir'])

        start_time = datetime.datetime.now()

        # Print Parameters and command
        with open(experiment_folder + '/' + setup_filename, 'w+') as file:
            print('', file=file)
            print('StartTime: %s' % start_time, file=file)
            print('Githash: %s' % git_hash, file=file)
            print('Parameters:', file=file)
            print(json.dumps(p, indent=4), file=file)
            print('Command: %s' % command, file=file)

        # Execute training
        print('Training started...')
        with open(experiment_folder + '/' + console_output_filename, 'w+') as file:

            command = command.split(" ")
            command.append("--start-from-round")
            command.append("") # Placeholder to be set to the round below

            step = 10
            start = p['start_round']
            for i in range(start, p['num_rounds'], step):
                end = min(i+step, p['num_rounds'])

                command[-1] = str(start)
                command[8] = str(end)

                print(f"Running {start} to {end}...")
                training = subprocess.Popen(command, stdout=file, stderr=file)
                training.wait()

                if training.returncode != 0:
                    raise Exception('Training subprocess failed')

                start = end

        # Document end of training
        print('Training finished. Documenting results...')
        with open(experiment_folder + '/' + setup_filename, 'a+') as file:
            end_time = datetime.datetime.now()
            print('EndTime: %s' % end_time, file=file)
            print('Duration Training: %s' % (end_time - start_time), file=file)

        print('Analysing tangle...')
        os.makedirs(experiment_folder + '/tangle_analysis', exist_ok=True)
        analysator = TangleAnalysator(experiment_folder + '/tangle_data', p['num_rounds'] - 1, experiment_folder + '/tangle_analysis')
        analysator.save_statistics(include_reference_statistics=(params['publish_if_better_than'] is 'REFERENCE'))

if __name__ == "__main__":
    main()

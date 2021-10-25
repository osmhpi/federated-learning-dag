import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .args import parse_args

from . import Lab, Dataset, TipSelectorFactory
from .config import LabConfiguration, ModelConfiguration, PoisoningConfiguration, RunConfiguration, NodeConfiguration, TipSelectorConfiguration

def main():
    run_config, lab_config, model_config, node_config, tip_selector_config, poisoning_config = \
        parse_args(RunConfiguration, LabConfiguration, ModelConfiguration, NodeConfiguration, TipSelectorConfiguration, PoisoningConfiguration)

    tip_selector_factory = TipSelectorFactory(tip_selector_config)
    lab = Lab(tip_selector_factory, lab_config, model_config, node_config, poisoning_config)

    dataset = Dataset(lab_config, model_config)

    lab.train(run_config.clients_per_round, run_config.start_from_round, run_config.num_rounds, run_config.eval_every, run_config.eval_on_fraction, dataset)
    #lab.print_validation_results(lab.validate(run_config.num_rounds-1, dataset, run_config.eval_on_fraction), mode='all')

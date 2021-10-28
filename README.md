# Federated Learning DAG Experiments

This repository contains software artifacts to reproduce the experiments presented in the Middleware '21 paper "Implicit Model Specialization through DAG-based Decentralized Federated Learning"

## General Usage

Use pipenv to set up your environment.

There are two variants of 'labs': A default, single-threaded one, and an extended version using the 'ray' parallelism library.

Basic usage: `python -m tangle.lab --help` (or `python -m tangle.ray --help`).

To execute experiments, change parameters in `experiments.py`. Then run `python experiments.py`.
Find available options by executing `python experiments.py --help`.

For executing a single step (i.e. a particular client training on local data and submitting a transaction), run `step.py f0000_14 10`, where `f0000_14` is the client id and `10` corresponds to `tangle_data/tangle_10.json`.

To view a DAG (sometimes called a tangle) in a web browser, run `python -m http.server` in the repository root and open [http://localhost:8000/viewer/](http://localhost:8000/viewer/). You may need to slide the slider to the left to see something.


## Reproduction of the evaluation in the paper

The experiements in the paper can be reproduced by running python scripts in the root folder of this repository. They are organized by the figures in which the respective evaluation is presented and named `experiments_figure_[*].py`

The results of the federated averaging runs presented in Figure 9 as baseline can be reproduced by running `run_fed_avg_[fmnist,poets,cifar].py` 
The results presented in Table 2 are generated by the scripts for DAG-IS of Figure 9 as well. 

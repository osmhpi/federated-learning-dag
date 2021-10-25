# Federated Learning DAG Experiment

Use pipenv to set up your environment.

There are two variants of 'labs': A default, single-threaded one, and an extended version using the 'ray' parallelism library.

Basic usage: `python -m tangle.lab --help` (or `python -m tangle.ray --help`).

To execute experiments, change parameters in `experiments.py`. Then run `python experiments.py`.
Find available options by executing `python experiments.py --help`.

For executing a single step (i.e. a particular client training on local data and submitting a transaction), run `step.py f0000_14 10`, where `f0000_14` is the client id and `10` corresponds to `tangle_data/tangle_10.json`.

To view tangles in a web browser, run `python -m http.server` in the repository root and open [http://localhost:8000/viewer/](http://localhost:8000/viewer/). You may need to slide the slider to the left to see something.

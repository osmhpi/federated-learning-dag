class LabConfiguration:
    seed: int
    model_data_dir: str
    tangle_dir: str
    src_tangle_dir: str

    def __init__(self):
        self.seed = None
        self.model_data_dir = None
        self.tangle_dir = None
        self.use_val_set = None

    def define_args(self, parser):
        parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
        parser.add_argument('--model-data-dir',
                    help='dir for model data',
                    type=str,
                    default='data',
                    required=False)
        parser.add_argument('--tangle-dir',
                    help='dir for tangle data (DAG JSON)',
                    type=str,
                    default='tangle_data',
                    required=False)
        parser.add_argument('--src-tangle-dir',
                    help='dir to load initial tangle data from (DAG JSON)',
                    type=str,
                    default=None,
                    required=False)
        parser.add_argument('--use-val-set',
                    help='use validation set',
                    action='store_true')

    def parse(self, args):
        self.seed = args.seed
        self.model_data_dir = args.model_data_dir
        self.tangle_dir = args.tangle_dir
        self.src_tangle_dir = args.src_tangle_dir
        self.use_val_set = args.use_val_set

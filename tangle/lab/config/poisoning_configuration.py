from ...core.poison_type import PoisonType
import argparse

class PoisoningConfiguration:

    def define_args(self, parser):
        POISON_TYPES = ['disabled', 'random', 'labelflip']

        parser.add_argument('--poison-type',
                    help='type of malicious clients considered',
                    type=str,
                    choices=POISON_TYPES,
                    default='disabled',
                    required=False)
        parser.add_argument('--poison-fraction',
                        help='fraction of clients being malicious',
                        type=float,
                        default=0,
                        required=False)
        parser.add_argument('--poison-from',
                        help='epoch to start poisoning from',
                        type=float,
                        default=1,
                        required=False)
        parser.add_argument('--poison-use-random-ts',
                    help='malicious nodes always use the random tip selector',
                    type=str2bool,
                    default=False)

    def parse(self, args):
        self.poison_type = {
            'disabled': PoisonType.Disabled,
            'random': PoisonType.Random,
            'labelflip': PoisonType.LabelFlip
        }[args.poison_type]
        self.poison_fraction = args.poison_fraction
        self.poison_from = args.poison_from
        self.use_random_ts = args.poison_use_random_ts

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

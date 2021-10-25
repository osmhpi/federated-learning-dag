import argparse

class TipSelectorConfiguration:

    def define_args(self, parser):
        parser.add_argument('--tip-selector',
                        help='tip selection algorithm',
                        type=str,
                        choices=['default', 'accuracy', 'lazy_accuracy'],
                        default='default')

        # Parameters for AccuracyTipSelector
        parser.add_argument('--acc-tip-selection-strategy',
                        help='strategy how to select the next tips',
                        type=str,
                        choices=['WALK', 'GLOBAL'],
                        default='WALK')

        parser.add_argument('--acc-cumulate-ratings',
                        help='whether after calculating accuracies should be cumulated',
                        type=str2bool,
                        default=False)

        parser.add_argument('--acc-ratings-to-weights',
                        help='algorithm to generate weights from ratings. Has effect only if used with WALK',
                        type=str,
                        choices=['LINEAR', 'ALPHA'],
                        default='LINEAR')

        parser.add_argument('--acc-select-from-weights',
                        help='algorithm to select the next transaction from given weights. Has effect only if used with WALK',
                        type=str,
                        choices=['MAXIMUM', 'WEIGHTED_CHOICE'],
                        default='MAXIMUM')

        parser.add_argument('--acc-alpha',
                        help='exponential factor for ratings to weights. Has effect only if used with WALK and WEIGHTED_CHOICE',
                        type=float,
                        default=0.001)

        # Parameters for particles
        parser.add_argument('--use-particles',
                        help='use particles to start walk instead of starting from genesis',
                        type=str2bool,
                        default=False)
        
        parser.add_argument('--particles-depth-start',
                        help='the begin of the depth based interval too choose particles from',
                        type=int,
                        default=10)
        
        parser.add_argument('--particles-depth-end',
                        help='the end of the depth based interval too choose particles from',
                        type=int,
                        default=20)

        parser.add_argument('--particles-number',
                        help='the number of particles to use. If num-tips < particles-number, only num-tips particles will be used',
                        type=int,
                        default=10)


    def parse(self, args):
        self.tip_selector = args.tip_selector
        self.use_particles = args.use_particles
        self.particles_depth_start = args.particles_depth_start
        self.particles_depth_end = args.particles_depth_end
        self.particles_number = args.particles_number
        self.acc_tip_selection_strategy = args.acc_tip_selection_strategy
        self.acc_cumulate_ratings = args.acc_cumulate_ratings
        self.acc_ratings_to_weights = args.acc_ratings_to_weights
        self.acc_select_from_weights = args.acc_select_from_weights
        self.acc_alpha = args.acc_alpha

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

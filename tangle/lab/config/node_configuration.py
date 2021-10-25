from ...core.node import NodeConfiguration as NC

class NodeConfiguration(NC):

    def define_args(self, parser):
        parser.add_argument('--num-tips',
                        help='number of tips being selected per round',
                        type=int,
                        default=2,
                        required=False)

        parser.add_argument('--sample-size',
                        help='number of possible tips being sampled per round',
                        type=int,
                        default=2,
                        required=False)

        parser.add_argument('--publish-if-better-than',
                        help='condition for publishing the transaction',
                        type=str,
                        choices=['REFERENCE', 'PARENTS'],
                        default='PARENTS')

        parser.add_argument('--reference-avg-top',
                        help='number models to average when picking reference model',
                        type=int,
                        default=1,
                        required=False)

    def parse(self, args):
        self.num_tips = args.num_tips
        self.sample_size = args.sample_size
        self.reference_avg_top = args.reference_avg_top
        self.publish_if_better_than = args.publish_if_better_than

class RunConfiguration:

    def define_args(self, parser):
        SIM_TIMES = ['small', 'medium', 'large']

        parser.add_argument('--num-rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
        parser.add_argument('--eval-every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
        parser.add_argument('--eval-on-fraction',
                        help='evaluate on fraction of clients;',
                        type=float,
                        default=0.1)
        parser.add_argument('--clients-per-round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
        parser.add_argument('--start-from-round',
                        help='at which round to start/resume training',
                        type=int,
                        default=0)
        parser.add_argument('-t',
                        help='simulation time: small, medium, or large;',
                        type=str,
                        choices=SIM_TIMES,
                        default='large')
        parser.add_argument('--target-accuracy',
                        help='stop training after reaching this test accuracy',
                        type=float,
                        default=1,
                        required=False)

    def parse(self, args):
        self.num_rounds = args.num_rounds
        self.eval_every = args.eval_every
        self.eval_on_fraction = args.eval_on_fraction
        self.clients_per_round = args.clients_per_round
        self.start_from_round = args.start_from_round
        self.t = args.t
        self.target_accuracy = args.target_accuracy

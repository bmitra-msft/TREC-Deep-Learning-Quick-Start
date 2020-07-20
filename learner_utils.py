class LearnerUtils:

    def __init__(self, printer):
        self.printer = printer

    def parser_add_args(self, parser):
        parser.add_argument('--seed', default=0, help='random seed (default: 0)', type=int)
        parser.add_argument('--device', default='cuda', help='device identifier (default: cuda)')
        parser.add_argument('--loss', default='ranknet', help='training loss (default: ranknet, also allowed: smoothmrr)')
        parser.add_argument('--lr', default=0.0001, help='learning rate (default: 0.0001)', type=float)
        parser.add_argument('--mb_size_train', default=32, help='minibatch size for training (default: 32)', type=int)
        parser.add_argument('--mb_size_test', default=256, help='minibatch size for test (default: 256)', type=int)
        parser.add_argument('--mb_size_infer', default=16, help='minibatch size for inference (default: 16)', type=int)
        parser.add_argument('--num_rand_negs', default=2, help='number of random negative documents for training (default: 2)', type=int)
        parser.add_argument('--epoch_size', default=4096, help='epoch size (default: 4096)', type=int)
        parser.add_argument('--num_epochs_train', default=32, help='number of epochs to train (default: 32)', type=int)
        parser.add_argument('--max_metric_pos', default=10, help='rank cutoff for computing discounted metrics (default: 10)', type=int)
        parser.add_argument('--max_metric_pos_nodisc', default=100, help='rank cutoff for computing non-discounted metrics (default: 100)', type=int)
        parser.add_argument('--file_out_model', default='model-{}.pt', help='filename for trained model (default: model-{}.pt)')
        parser.add_argument('--file_out_scores', default='scores-{}-{}.txt', help='filename for eval run file in TREC submission format (default: scores-{}-{}.txt)')
        parser.add_argument('--orcas_train', help='use orcas data for model training', action='store_true')
        parser.add_argument('--single_gpu', help='disable multi GPU training', action='store_true')

    def parser_validate_args(self, args):
        self.args = args
        assert args.mb_size_train > 0 and args.mb_size_test > 0, 'minibatch size must be greater than zero'
        assert args.epoch_size > 0, 'epoch size must be greater than zero'
        assert args.num_epochs_train > 0, 'number of epochs must be greater than zero'
        assert args.device == 'cuda' or not args.single_gpu, 'can not disable multi gpu training when device is not set to cuda'

    def setup_and_verify(self):
        pass

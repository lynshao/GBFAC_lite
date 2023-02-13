import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Sequence arguments
    parser.add_argument('--snr1', type=float, default= 1, help="Transmission SNR")
    parser.add_argument('--snr2', type=float, default= 100., help="Feedback SNR")
    parser.add_argument('--K', type=int, default=51, help="Sequence length")
    parser.add_argument('--block_size', type=int, default=3, help="Block size")
    parser.add_argument('--numb_block', type=int, default=17, help="Number of blocks")
    parser.add_argument('--parity_pb', type=int, default=6, help="Number of parity bits")
    parser.add_argument('--seq_reloc', type=int, default=1)
    parser.add_argument('--memory', type=int, default=51)
    parser.add_argument('--core', type=int, default=8)

    # Transformer arguments
    parser.add_argument('--heads_trx', type=int, default=1, help="number of heads for the multi-head attention")
    parser.add_argument('--d_k_trx', type=int, default=32, help="number of features for each head")
    parser.add_argument('--N_trx', type=int, default=2, help=" number of layers in the encoder and decoder")
    parser.add_argument('--dropout', type=float, default=0.0, help="prob of dropout")
    parser.add_argument('--custom_attn', type=bool, default = True, help= "use custom attention")
    parser.add_argument('--start', type=int, default=0)

    # Learning arguments
    parser.add_argument('--load_weights') # None
    parser.add_argument('--train', type=int, default= 1)
    parser.add_argument('--reloc', type=int, default=1, help="w/ or w/o power rellocation")
    parser.add_argument('--totalbatch', type=int, default=60, help="number of total batches to train; scale it with 10k")
    parser.add_argument('--batchSize', type=int, default=8192, help="batch size")
    parser.add_argument('--opt_method', type=str, default='adamW', help="Optimization method adamW,lamb,adam")
    parser.add_argument('--clip_th', type=float, default=0.5, help="clipping threshold")
    parser.add_argument('--use_lr_schedule', type=bool, default = True, help="lr scheduling")
    parser.add_argument('--multclass', type=bool, default = True, help="bit-wise or class-wise training")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--wd', type=float, default=0.01, help="weight decay")
    args = parser.parse_args()

    return args

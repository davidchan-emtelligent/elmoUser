import os
import sys
import json
import argparse
import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from helper import get_tokens_count

def main(args):

    # define the options
    config_file = args.config_file
    if config_file == None:
        config_file = os.path.join(current_dir, "resources/default_config.json")
    with open(config_file, "r") as fj:
        options = json.load(fj)

    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # number of tokens in training data (this for 1B Word Benchmark)
    # batch_no = n_epochs*n_train_tokens/(batch_size*unroll_steps*n_gpus)
    #25600  => 100 n_batch  #example filtered 1330337  #1B 768648884
    if 'n_train_tokens' not in options:
        options['n_train_tokens'] = get_tokens_count(args.train_prefix)
    else:
        print("Warning: using options['n_train_tokens']:", options['n_train_tokens'])  

    n_gpus = options['n_gpus']
    options['n_tokens_vocab'] = vocab.size
    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False, shuffle_on_load=True)
    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir

    print("options:", options)
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--config_file', default=None, help='Config.json file')

    args = parser.parse_args()
    main(args)


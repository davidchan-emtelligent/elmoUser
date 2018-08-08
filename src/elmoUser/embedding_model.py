import os
import sys
import argparse

from bilm.training import dump_weights as dw

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from helper import load_options, save_options

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--out_dir', help='Location to keep vocabs, weights and options')

    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.system("mkdir %s"%args.out_dir)

    dw(args.save_dir, os.path.join(args.out_dir, 'weights.hdf5'))

    os.system("cp %s/vocabs.txt %s/"%(args.save_dir, args.out_dir))

    options = load_options(os.path.join(args.save_dir, 'options.json'))
    # fix "n_characters" in options.json
    options['char_cnn']["n_characters"] += 1
    save_options(options, os.path.join(args.out_dir, 'options.json'))

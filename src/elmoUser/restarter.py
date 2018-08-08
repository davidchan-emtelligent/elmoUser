import os
import sys
import argparse
import numpy as np
import tensorflow as tf

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from helper import get_tokens_count, clean_checkpoint, load_options, save_options

def resume(options, prefix, vocab, n_gpus, tf_save_dir, tf_log_dir, ckpt_file):
	kwargs = {
		'test': False,
		'shuffle_on_load': True,
	}
	tf.reset_default_graph()
	if options.get('bidirectional'):
		data = BidirectionalLMDataset(prefix, vocab, **kwargs)
	else:
		data = LMDataset(prefix, vocab, **kwargs)

	train(options, data, n_gpus, tf_save_dir, tf_log_dir, restart_ckpt_file=ckpt_file)
	clean_checkpoint(tf_save_dir)


def top_level(args):
	options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

	if 'char_cnn' in options:
		max_word_length = options['char_cnn']['max_characters_per_token']
	else:
		max_word_length = None

	vocab = load_vocab(os.path.join(args.save_dir, "vocabs.txt"), max_word_length)

	tf_save_dir = args.save_dir
	tf_log_dir = args.save_dir

	# set optional inputs to overide the otpions.json
	if args.n_epochs > 0:
		options['n_epochs'] = args.n_epochs
	if args.batch_size > 0:
		options['batch_size'] = args.batch_size
	if args.n_gpus > 0:
		n_gpus = args.n_gpus
	else:
		n_gpus = options['n_gpus']

	# load train_prefixes
	#if args.train_prefix_paths != None:
	if False:
		with open(args.train_prefix_paths, "r") as fd:
			train_prefixes = fd.read().split('\n')
		train_prefixes = [f for f in train_prefixes if f != ""]
		options['train_prefix_paths'] = train_prefixes
		start = 0
	else:
		train_prefixes = options['train_prefix_paths']
		start = options['milestone']

	if start >= len(train_prefixes):
		print("WARNING: Finish all train_prefix_paths. Reset milestone in options.")
		sys.exit(0)

	# loop all train_prefix_paths
	milestone = start
	for train_prefix in train_prefixes[start:]:
		prefix = train_prefix + '/*'

		if args.n_train_tokens > 0:
			options['n_train_tokens'] = args.n_train_tokens
		else:
			options['n_train_tokens'] =  get_tokens_count(prefix)

		resume(options, prefix, vocab, n_gpus, tf_save_dir, tf_log_dir, ckpt_file)
		milestone += 1
		options['milestone'] = milestone
		save_options(options, os.path.join(args.save_dir, "options.json"))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_dir', default=None, help='Location of checkpoint files')
	#parser.add_argument('--train_prefix_paths', default=None, help='Prefix paths for train files')
	parser.add_argument('--n_gpus', type=int, default=0, help='Number of GPUs to use')
	parser.add_argument('--batch_size', type=int, default=0)
	parser.add_argument('--n_train_tokens', type=int, default=0)
	parser.add_argument('--n_epochs', type=int, default=0)

	args = parser.parse_args()

	if args.save_dir == None:
		print("ERROR: no save_dir")
	else:
		top_level(args)


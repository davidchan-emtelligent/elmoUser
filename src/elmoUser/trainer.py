import os
import sys
import json
import argparse
import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from helper import get_tokens_count, load_options, save_options
import restarter

def top_level(args):
	if not os.path.isdir(args.save_dir):
		os.system("mkdir %s"%args.save_dir)

	# define the options
	if args.config_file == None:
		args.config_file = os.path.join(current_dir, "resources/default_config.json")
	options = load_options(args.config_file)

	# load train_prefixes
	with open(args.train_prefix_paths, "r") as fd:
		train_prefixes = fd.read().split('\n')
	train_prefixes = [f for f in train_prefixes if f != ""]
	options['train_prefix_paths'] = train_prefixes

	# load the vocab
	vocab = load_vocab(args.vocab_file, 50)

	# number of tokens in training data (this for 1B Word Benchmark)
	# batch_no = n_epochs*n_train_tokens/(batch_size*unroll_steps*n_gpus)
	#25600  => 100 n_batch  #example filtered 1330337  #1B 768648884
	if args.n_train_tokens == None:
		options['n_train_tokens'] = get_tokens_count(args.train_prefix)
	else:
		options['n_train_tokens'] = args.n_train_tokens  

	options['n_tokens_vocab'] = vocab.size
	options['milestone'] = 0
	os.system("cp %s %s/vocabs.txt"%(args.vocab_file, args.save_dir))

	n_gpus = options['n_gpus']
	tf_save_dir = args.save_dir
	tf_log_dir = args.save_dir

	prefix = train_prefixes[0] + '/*'
	data = BidirectionalLMDataset(prefix, vocab, test=False, shuffle_on_load=True)

	print("options:", options)
	train(options, data, n_gpus, tf_save_dir, tf_log_dir)
	options['milestone'] = 1
	save_options(options, os.path.join(args.save_dir, "options.json"))

	if len(train_prefixes) == 1:
		return

	options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

	# loop all train_prefix_paths
	milestone = 1
	for train_prefix in train_prefixes[1:]:
		prefix = train_prefix + '/*'

		if args.n_train_tokens > 0:
			options['n_train_tokens'] = args.n_train_tokens
		else:
			options['n_train_tokens'] =  get_tokens_count(prefix)

		restarter.resume(options, prefix, vocab, n_gpus, tf_save_dir, tf_log_dir, ckpt_file)
		milestone += 1
		options['milestone'] = milestone
		save_options(options, os.path.join(args.save_dir, "options.json"))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_dir', default=None, help='Location of checkpoint files')
	parser.add_argument('--vocab_file', default=None, help='Vocabulary file')
	parser.add_argument('--train_prefix_paths', default=None, help='Prefixes for train files')
	parser.add_argument('--config_file', default=None, help='Config.json file')
	parser.add_argument('--n_train_tokens', default=None, type=int, help='optional: n_train_tokens for testing')

	args = parser.parse_args()

	if args.save_dir == None or args.vocab_file == None or args.train_prefix_paths == None:
		print("ERROR: no save_dir or vocab_file or train_prefix_paths")
	else:
		top_level(args)


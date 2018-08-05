import os
import multiprocessing

train_prefix = "/shared/dropbox/ctakes_conll/tokenized_text/ds_sentences_0"

def func(f):
	with open(f, "r") as fd:
		tokens= fd.read().strip().split()
	return len(tokens)

def get_tokens_count(train_prefix):

	fs = [os.path.join(train_prefix, f) for f in os.listdir(train_prefix)]
	ret_lst = multiprocessing.Pool().map(func, fs)

	return (sum(ret_lst))

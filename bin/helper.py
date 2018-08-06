import os
import multiprocessing

def clean_checkpoint(checkpoint_path):
	  fs = os.listdir(checkpoint_path)
	  with open(checkpoint_path + "/checkpoint", "r") as fd:
		lines = fd.read().split('\n')
	  valid_ckpt = lines[0].split(":")[-1].strip().replace('"', '')
	  print("valid checkpoint:", valid_ckpt)
	  for f in fs:
		if f == 'checkpoint' \
		or f.endswith('json') \
		or f.endswith('txt') \
		or f.endswith('hdf5') :
			continue
		if len(f.split(valid_ckpt)) == 1:
			run_str = "rm %s"%(os.path.join(checkpoint_path, f))
			print(run_str)
			os.system(run_str)

train_prefix = "/shared/dropbox/ctakes_conll/tokenized_text/ds_sentences_0"

def func(f):
	  with open(f, "r") as fd:
	  	  tokens= fd.read().strip().split()
	  return len(tokens)

def get_tokens_count(train_prefix):
	  train_prefix = train_prefix.replace("*", "")
	  fs = [os.path.join(train_prefix, f) for f in os.listdir(train_prefix)]
	  ret_lst = multiprocessing.Pool().map(func, fs)

	  return (sum(ret_lst))

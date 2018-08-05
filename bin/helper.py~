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
			#os.system(run_str)

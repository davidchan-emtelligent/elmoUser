import os
import multiprocessing

def clean_checkpoint(checkpoint_path):
    fs = os.listdir(checkpoint_path)
    with open(checkpoint_path + "/checkpoint", "r") as fd:
        lines = fd.read().split('\n')
    valid_chpt = lines[0].split(":")[-1].strip().replace('"', '')
    for f in fs:
	if f == 'checkpoint' \
	or f.endswith('json') \
        or f.endswith('txt') \
        or f.endswith('hdf5') :
            continue
        if len(f.split(valid_ckpt)) == 1:
            print(os.path.join(checkpoint_path, f))

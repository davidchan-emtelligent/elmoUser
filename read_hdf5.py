import h5py
#filename = 'tests/fixtures/model/lm_weights.hdf5'
filename = '../model/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_keys = list(f.keys()); print (a_group_keys)

# Get the data
for key in a_group_keys:
	data = list(f[key])
	print(key,  len(data), ":\n", data[0])


import h5py

#filename = 'tests/fixtures/model/lm_weights.hdf5'
filename = 'elmo_token_embeddings.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_keys = list(f.keys()); print (a_group_keys)

# Get the data
for key in a_group_keys:
	data = list(f[key])
	print(key,  len(data), ":\n", data)

vocab_file = 'vocab_small.txt'
with open(vocab_file, 'r') as fd:
	vocabs = fd.read().split('\n')

for (w, v) in zip(vocabs, list(f['embedding'])):
	print (w, ":", v)

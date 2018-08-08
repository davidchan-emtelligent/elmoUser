## elmoUser

Using embedding from elmo https://github.com/allenai/bilm-tf . All codes are modified from elmo/bilm-tf.

<br>

### Installation:

	virtualenv -p python3 venv3

	source venv3/bin/activate

	pip install -e .

<br>

### Test Installation:

	python -m unittest discover tests/

<br>

### Train:
	export CUDA_VISIBLE_DEVICES=0,1

	python bin/train_elmo.py \
	--train_prefix='tests/data/training_filtered/*' \
	--vocab_file tests/data/vocab_filtered.txt \
	--save_dir checkpoint \
	--config_file bin/resources/small_config.json

<br>

### Test:

	python bin/run_test.py \
	--test_prefix='tests/data/heldout_filtered/*' \
	--save_dir checkpoint

<br>

### Retrain:

	python bin/restart.py \
	--train_prefix='tests/data/training_filtered/*' \
	--save_dir checkpoint
	
Auto retrain a sequence of training dirs with span

	python bin/run_restart.py  \
	--save_dir checkpoint \
	--prefixes_dir tests/data/training_dir.paths \
	--span 1:4

<br>

### Get weights:

	python bin/dump_weights.py \
	--save_dir checkpoint \
	--outfile checkpoint/weights.hdf5

<br>

### Get vectors:
Provide a list of tokenized sentences.

	python bin/elmo_embedding.py \
	--save_dir checkpoint \
	--input_text tests/data/tokenized_sentences.txt	
	
Or:

	from bilm import ElmoEmbedding
	elmo = ElmoEmbedding(save_dir)
	elmo_context_vecs, context_tokens, context_ids = elmo(tokenized_sentences)
	
<br>

### Args:

1 vocab_file: 

	vocabs.txt (will be saved in save_dir and not be changed)

2 train_prefix: 

	dir/* (train all files in dir/)

3 save_dir:

	checkpoint (save model, vocabs.txt, options.json, weights.hdf5)
	model is no needed to get embedding vectors.

In options.json:

	batch_no = n_epochs*n_batches_per_epoch 

	n_batches_per_epoch = n_train_tokens/(batch_size*unroll_steps*n_gpus)

	epoch = batch_no/n_batches_per_epoch









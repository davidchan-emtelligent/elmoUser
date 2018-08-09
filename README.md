## elmoUser

Using embedding from elmo https://github.com/allenai/bilm-tf . All codes are modified from elmo/bilm-tf.

<br>

### Installation:

	virtualenv -p python3 venv3

	source venv3/bin/activate
	
	pip install path\_to\_bilm-tf
	
	pip install -e .

<br>

### Train:
	export CUDA_VISIBLE_DEVICES=0,1

	elmo_trainer \
	--train_prefix_paths tests/data/train_prefixes.paths \
	--vocab_file tests/data/vocab_data.txt \
	--save_dir checkpoint \
	--config_file src/elmoUser/resources/small_config.json \
	--n_train_tokens 1280

<br>

### Test:

	elmo_tester \
	--test_prefix='tests/data/heldout_data/*' \
	--save_dir checkpoint

<br>

### Retrain:

	elmo_restarter \
	--save_dir checkpoint

<br>

### Embedding model:

	elmo_model \
	--save_dir checkpoint \
	--out_dir elmo_embedding_model

<br>

### Embedding vectors:
Provide a list of tokenized sentences.

	python src/elmoUser/embedding.py \
	--model_path elmo_embedding_model \
	--input_path tests/data/training_data/0_5000.txt	
	
Or:

	from bilm import ElmoEmbedding
	elmo = ElmoEmbedding(model_path)
	elmo_context_vecs, context_tokens, context_ids = elmo(tokenized_sentences)
	
<br>

### Args:

1 vocab_file: 

	vocabs.txt (will be saved in save_dir).

2 train_prefix_paths: 

	txt file to keep all prefixs paths.

3 save_dir:

	checkpoint (checkpoint, models, vocabs.txt, options.json, weights.hdf5).
	
3 model_path:

	emdedding model (vocabs.txt, options.json, weights.hdf5).
	
where options\['char_cnn']\["n_characters"] += 1 in embedding_model
    
In options.json:

	batch_no = n_epochs*n_batches_per_epoch 

	n_batches_per_epoch = n_train_tokens/(batch_size*unroll_steps*n_gpus)

	epoch = batch_no/n_batches_per_epoch
	
You can use --n_train_tokens in trainer and restarter to limit the no. of samples when debugging (12800 or 25600).









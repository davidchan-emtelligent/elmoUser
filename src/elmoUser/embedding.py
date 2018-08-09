'''
ELMo use character inputs.

    python src/elmoUser/embedding.py --model_path elmo_embedding_model -input_path tests/data/training_data/0_5000.txt

'''
import os
import sys
import json
import argparse
import numpy as np

import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class ElmoEmbedding:
    def __init__(self, model_path):
        vocab_file = os.path.join(model_path, 'vocabs.txt')
        options_file = os.path.join(model_path, 'options.json')
        weight_file = os.path.join(model_path, 'weights.hdf5')
        with open(options_file, "r") as fj:
            options = json.load(fj)
        self.max_characters_per_token = options['char_cnn']['max_characters_per_token']        

        # Create a Batcher to map text to character ids.
        self.batcher = Batcher(vocab_file, self.max_characters_per_token)
        # Build the biLM graph.
        self.bilm = BidirectionalLanguageModel(options_file, weight_file)


    def __call__(self, tokenized_sentences_lst):
        # Input placeholders to the biLM.
        context_character_ids = tf.placeholder('int32', shape=(None, None, self.max_characters_per_token))

        # Get ops to compute the LM embeddings.
        context_embeddings_op = self.bilm(context_character_ids)

        # Get an op to compute ELMo (weighted average of the internal biLM layers)
        elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
        elmo_context_output = weight_layers('output', context_embeddings_op, l2_coef=0.0)

        # Now we can compute embeddings.
        context_tokens  = [sentence.split() for sentence in tokenized_sentences_lst]

        with tf.Session() as sess:
            # It is necessary to initialize variables once before running inference.
            sess.run(tf.global_variables_initializer())

            # Create batches of data.
            context_ids = self.batcher.batch_sentences(context_tokens)

            # Compute ELMo representations (here for the input only, for simplicity).
            elmo_context_vecs = sess.run(
            [elmo_context_input['weighted_op']],
            feed_dict={context_character_ids: context_ids}
            )

        return elmo_context_vecs[0]  #, context_tokens, context_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default = None, help = 'model_path with weights')
    parser.add_argument('--input_path', default = '', help = 'input tokenized text')

    args = parser.parse_args()

    if args.model_path == None:
        print("ERROR: no save_dir")
        sys.exit(0)

    if args.input_path == "":
        tokenized_sentences = ['Pretrained biLMs compute representations useful for NLP tasks .', \
                'They give state of the art performance for many tasks .']
    else:
        with open(args.input_path, "r") as fd:
            tokenized_sentences = fd.read().split('\n')
        tokenized_sentences = [sentence for sentence in tokenized_sentences if sentence != ""][:128]

    elmo = ElmoEmbedding(args.model_path)
    elmo_context_vecs = elmo(tokenized_sentences)
    #elmo_context_vecs, context_tokens, context_ids = elmo(tokenized_sentences)

    if args.input_path == "":
        print ("\nlengths:",[len(sent.split()) for sent in tokenized_sentences[:50]], "\n")
        print ("tokenized sentences:")
        print ("\n".join(tokenized_sentences), "\n")
        print ("tokens:")
        print ("\n".join([str(sent.split()) for sent in tokenized_sentences]), "\n")
        print ("vecs.shape:", elmo_context_vecs.shape, "\n")
        #print ("context_ids.shape:", context_ids.shape)
        #print ("context_ids:", context_ids)
        """
        lengths: [9, 11] 

        tokenized sentences:
        Pretrained biLMs compute representations useful for NLP tasks .
        They give state of the art performance for many tasks . 

        tokens:
        ['Pretrained', 'biLMs', 'compute', 'representations', 'useful', 'for', 'NLP', 'tasks', '.']
        ['They', 'give', 'state', 'of', 'the', 'art', 'performance', 'for', 'many', 'tasks', '.'] 

        vecs.shape: (2, 11, 1024)

        context_ids: (2, 13, 50) 
         [[[259 257 260 ... 261 261 261]   1, '<S>'
          [259  81 115 ... 261 261 261]    2, 'Pretrained'
          [259  99 106 ... 261 261 261]    3, 'biLMs'
          ...
          [259 258 260 ... 261 261 261]    11, '</S>'
          [  0   0   0 ...   0   0   0]    12
          [  0   0   0 ...   0   0   0]]   13

         [[259 257 260 ... 261 261 261]    1, '<S>'
          [259  85 105 ... 261 261 261]    2, 'They'
          [259 104 106 ... 261 261 261]    3, 'give'
          ...
          [259 117  98 ... 261 261 261]    11, 'tasks'
          [259  47 260 ... 261 261 261]    12, '.'
          [259 258 260 ... 261 261 261]]]  13, '</S>'
        """
    else:
        print ("\nlengths:",[len(sent.split()) for sent in tokenized_sentences[:50]], "\n")
        print ("tokenized sentences:")
        print ("\n".join(tokenized_sentences[:2]), "\n")
        print ("tokens:")
        print ("\n".join([str(sent.split()) for sent in tokenized_sentences[:2]]), "\n")
        print ("vecs.shape:", elmo_context_vecs.shape)


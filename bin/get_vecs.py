'''
ELMo use character inputs.

python get_vecs.py -c nlp/tests/fixtures/model -v nlp/tests/fixtures/model/vocab_test.txt

python get_vecs.py -c checkpoint -v ctakes_vocabs.txt -i text.txt

'''
import os
import sys
import json
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

MAX_CHAR_LEN = 50

def load_bilm(args):
    # Location of pretrained LM.  Here we use the test fixtures.
    chechpoint = args.checkpoint
    vocab_file = args.vocab_file
    options_file = os.path.join(chechpoint, 'options.json')
    weight_file = os.path.join(chechpoint, 'weights.hdf5')

    # fix "n_characters" in options.json
    with open(options_file, "r") as fj:
        options = json.load(fj)
    if options['char_cnn']["n_characters"] != 262:
        options['char_cnn']["n_characters"] = 262
    with open(options_file, "w") as fj:
        json.dump(options, fj)

    # Create a Batcher to map text to character ids.
    batcher = Batcher(vocab_file, MAX_CHAR_LEN)

    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(options_file, weight_file)

    return bilm, batcher


def get_vectors(tokenized_sentences_lst, bilm, batcher):
    # Input placeholders to the biLM.
    context_character_ids = tf.placeholder('int32', shape=(None, None, MAX_CHAR_LEN))

    # Get ops to compute the LM embeddings.
    context_embeddings_op = bilm(context_character_ids)

    # Get an op to compute ELMo (weighted average of the internal biLM layers)
    elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
    elmo_context_output = weight_layers('output', context_embeddings_op, l2_coef=0.0)

    # Now we can compute embeddings.
    context_tokens  = [sentence.split() for sentence in tokenized_sentences_lst]

    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        # Create batches of data.
        context_ids = batcher.batch_sentences(context_tokens )

        # Compute ELMo representations (here for the input only, for simplicity).
        elmo_context_vecs = sess.run(
        [elmo_context_input['weighted_op']],
        feed_dict={context_character_ids: context_ids}
        )

    return elmo_context_vecs[0], context_tokens, context_ids 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', dest = 'checkpoint', default = 'checkpoint', help = 'checkpoint with weights')
    parser.add_argument('-v', '--vocab_file', dest = 'vocab_file', help = 'vocabulary file')
    parser.add_argument('-i', '--input_file', dest = 'input_file', default = "", help = 'input tokenized text')

    args = parser.parse_args()

    if args.input_file == "":
        tokenized_sentences = [
            'Pretrained biLMs compute representations useful for NLP tasks .',
            'They give state of the art performance for many tasks .'
        ]
    else:
        with open(args.input_file, "r") as fd:
            tokenized_sentences = fd.read().split('\n')
        tokenized_sentences = (sentence for sentence in tokenized_sentences if sentence != "")

    bilm, batcher = load_bilm(args)

    elmo_context_vecs, context_tokens, context_ids = get_vectors(tokenized_sentences, bilm, batcher)

    if args.input_file == "":
        print (len(context_tokens ),len(context_tokens [0]),len(context_tokens [1]),context_tokens )
        print (elmo_context_vecs.shape)
        print (context_ids.shape, '\n', context_ids)
        """
        2 9 11 
        [['Pretrained', 'biLMs', 'compute', 'representations', 'useful', 'for', 'NLP', 'tasks', '.'], 
        ['They', 'give', 'state', 'of', 'the', 'art', 'performance', 'for', 'many', 'tasks', '.']]

        (2, 11, 1024)                      word vec size = 1024

        (2, 13, 50) 
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
        print ("1st line  :", context_tokens[0])
        print ("ids.shape :", context_ids.shape)
        print ("vecs.shape:", elmo_context_vecs.shape)


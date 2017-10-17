"""
RNN.py
	A Dynet rnn network model. 
	Train a model and predict classes(tags:punctuations) for sequences of elements(features:chars/words).

Usage:

import RNN

#build a new RNN
#specs = [VOCAB_SIZE, EMBEDDINGS_SIZE, STATE_SIZE, NUM_OF_CLASSES, NUM_OF_LAYERS, rnn_cell, optimizer, model_path]
specs = [100, 25, 30, 2, 1, 'lstm', 'Adagrad', model/punctuation.model]
rnn = RNN(specs)
rnn.train(X, Y, X_dev, Y_dev, batch_size, max_iteration, display_every)
y_pred_lst = rnn.predict(x_lst)
rnn.save(path)

#load from path
rnn1 = RNN()
rnn1.load(path)
rnn1.train(X, Y, X_dev, Y_dev, batch_size, max_iteration, display_every)
y_pred_lst = rnn1.predict(x_lst)
mlp1.save(other_path)
"""

import _dynet as dn
dyparams = dn.DynetParams()
dyparams.from_args()
dyparams.set_mem(4096)
dyparams.set_random_seed(801)
dyparams.init() # or init_from_params(dyparams)
from dynet import *

from numpy import mean, argmax
import sys

class RNN(object):
	#create a RNN object	
	def __init__(self, spec=[]):
		self.model = dn.Model()
		if spec != []:
			self.initial(spec)

	#initial a rnn
	def initial(self, spec):
		[VOCAB_SIZE, EMBEDDINGS_SIZE, HIDDEN_SIZE, NUM_OF_CLASSES, NUM_OF_LAYERS, rnn_cell, \
			optimizer, model_path] = spec
		self.spec = spec 
		self.rnn_cell = rnn_cell
		self.optimizer = optimizer
		self.model_path = model_path

		if self.rnn_cell == 'GRU':
			self.rnn = dn.GRUBuilder(NUM_OF_LAYERS, EMBEDDINGS_SIZE, HIDDEN_SIZE, self.model)
		else:
			self.rnn = dn.LSTMBuilder(NUM_OF_LAYERS, EMBEDDINGS_SIZE, HIDDEN_SIZE, self.model)

		if self.optimizer == 'Adagrad':
			self.trainer = dn.AdagradTrainer(self.model)
		elif self.optimizer == 'SGD':
			self.trainer = dn.SimpleSGDTrainer(self.model)
		else:
			self.trainer = dn.RMSPropTrainer(self.model, rho=0.9)
			#self.trainer = dn.AdadeltaTrainer(self.model, rho= 0.9)

		self.input_lookup = self.model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
		self.W = self.model.add_parameters((NUM_OF_CLASSES, HIDDEN_SIZE))
		self.B = self.model.add_parameters((NUM_OF_CLASSES))
		
	
	#get probs for whole batch
	#probs.shape = (classes, samples)
	def get_probs(self, batch):
		dn.renew_cg()

		# The I iteration embed all the i-th items in all batches
		embedded = [dn.lookup_batch(self.input_lookup, chars) for chars in zip(*batch)]
		state = self.rnn.initial_state()
		output_vec = state.transduce(embedded)[-1]
		w = dn.parameter(self.W)
		b = dn.parameter(self.B)
		
		return w*output_vec+b
	
	
	#train data_lst with input batch_size
	def train(self, trainX, trainY, testx, testy, batch_size, max_iter=10, display_every=1):

		best_dev_acc = 0.0				
		for i in range(max_iter):
			ith = i + 1

			for batch_x, batch_y in self.generate_batch(trainX, trainY, batch_size):
				
				probs = self.get_probs(batch_x)
				loss = dn.sum_batches(dn.pickneglogsoftmax_batch(probs, batch_y))
				loss_value = loss.value()
				loss.backward()
				self.trainer.update()

			train_acc = self.validate(trainX, trainY, batch_size)

			y_pred = self.predict(testx, batch_size)
			dev_acc = sum(map(lambda (y, p): int(y == p), zip(testy, y_pred)))

			print "\riter:%d train_loss:%.4f train_acc: %.4f val_acc:%.4f  "%(ith, loss_value, \
				train_acc, float(dev_acc)/len(y_pred)),
			sys.stdout.flush()


			if dev_acc > best_dev_acc :   #and train_acc > 0.98:
				best_dev_acc = dev_acc
				self.save(self.model_path)
				print "\nupdate model:",self.model_path
			elif ith % display_every == 0:
				print			   
				
		return "done!"


	#validate data_lst with input batch_size
	def validate(self, trainX, trainY, batch_size):

		acc = []
		for batch_x, batch_y in self.generate_batch(trainX, trainY, batch_size):

			if len(batch_x) == 1:
				#problem with 1 row in get_probs()
				probs = self.get_probs(batch_x*2).npvalue()
				probs = probs[:,:1]
			else:
				probs = self.get_probs(batch_x).npvalue()

			for i in range(len(probs[0])):
				pred = argmax(probs[:, i])
				label = batch_y[i]
				if pred == label:
					acc.append(1)
				else:
					acc.append(0)
					
		return mean(acc)


	#predict a list [sample1, sample2, ....]
	def predict1(self, x_lst, batch_size=1):

		y_pred = []
		for i in xrange(0, len(x_lst), batch_size):	
			probs = self.get_probs(x_lst[i:i+batch_size]).npvalue()
			for i in range(len(probs[0])):

				y_pred += [argmax(probs[:, i])]

		return y_pred


	#predict a list [sample1, sample2, ....]
	def predict(self, x_lst, batch_size=2):

		y_pred = []
		for i in xrange(0, len(x_lst), batch_size):
			batch_x = x_lst[i:i+batch_size]	
			if len(batch_x) == 1:
				#problem with 1 row in get_probs()
				probs = self.get_probs(batch_x*2).npvalue()
				probs = probs[:,:1]
			else:
				probs = self.get_probs(batch_x).npvalue()

			for i in range(len(probs[0])):
				y_pred += [argmax(probs[:, i])]

		return y_pred


	#train data_batches
	def train_batches(self, batches_x, batches_y, testx, testy, max_iter=10, display_every=5):
		
		best_dev_acc = 0.0	
		for i in range(max_iter):
			ith = i + 1
			for batch_x, batch_y in zip(batches_x, batches_y):
				
				probs = self.get_probs(batch_x)
				loss = dn.sum_batches(dn.pickneglogsoftmax_batch(probs, batch_y))
				loss_value = loss.value()
				loss.backward()
				self.trainer.update()

			train_acc = self.validate_batches(batches_x, batches_y)

			y_pred = self.predict_list(testx)
			dev_acc = sum(map(lambda (y, p): int(y == p), zip(testy, y_pred)))

			print "\riter:%d train_loss:%.4f train_acc: %.4f val_acc:%.4f  "%(ith, loss_value, \
				train_acc, float(dev_acc)/len(y_pred)),
			sys.stdout.flush()

			if dev_acc > best_dev_acc:
				best_dev_acc = dev_acc
				self.save(self.model_path)
				print "\nupdate model:",self.model_path
			elif ith % display_every == 0:
				print		   
				
		return "done!"
	


	#validate batches
	def validate_batches(self, batches_x, batches_y):

		acc = []
		for batch_x, batch_y in zip(batches_x, batches_y):

			probs = self.get_probs(batch_x).npvalue()
			for i in range(len(probs[0])):
				pred = argmax(probs[:, i])
				label = batch_y[i]
				if pred == label:
					acc.append(1)
				else:
					acc.append(0)
					
		return 'train_acc: %.4f'%(mean(acc))

	
	
	#predict a list [sample1, sample2, ....]
	def predict_list(self, x_lst):

		y_pred = []
		probs = self.get_probs(x_lst).npvalue()
		for i in range(len(probs[0])):
			y_pred += [argmax(probs[:, i])]

		return y_pred
	
	
	# the Saveable interface requires the implementation
	# of the two following methods, specifying all the
	# Parameters / LookupParameters / LSTMBuilder / Saveables / etc
	# that are directly created by this Saveable.
	def get_components(self):
		return (dn.parameter(self.input_lookup).npvalue(), \
			dn.parameter(self.W).npvalue(), dn.parameter(self.B).npvalue())

	def restore_components(self, components):
		self.input_lookup, self.W, self.B = components


	#save rnn and specs				
	def save(self, path):
		
		#save the rnn and weigths
		self.model.save(path, [self.rnn, self.input_lookup, self.W, self.B])
		
		#save nn specification parameters
		spec = map(lambda x: "%d"%(x), self.spec[:-3]) + self.spec[-3:]
		with open(path +'.spec', 'w') as fd:
			fd.write('\n'.join(spec))

			
	#load a trained model and its weights		
	def load(self, path):

		#load the parameters 
		with open(path +'.spec', 'r') as fd:
			spec = fd.read().split('\n')
		spec = map(lambda x: int(x), spec[:-3]) +spec[-3:]
		
		#re-initial the MLP() with spec
		self.initial(spec)
		
		#load the rnn and it's weights
		self.rnn, self.input_lookup, self.W, self.B = self.model.load(path)


	#put into minibatches
	def generate_batch(self, X, Y, batch_size):   

		len1 = len(X)
		for i in xrange(0, len1, batch_size):

			batched_x = X[i:i+batch_size]
			batched_y = Y[i:i+batch_size]

			yield batched_x, batched_y


#testing
if __name__=="__main__":

	#line: "tag sequence|" => "y x"
	data = \
	". utropenic feverCommunity acqui|"+\
	". uired pneumoniaSevere sepsis##|"+\
	". ##Severe sepsisAnemia#########|"+\
	". #########AnemiaStage II breast|"+\
	". I breast cancerGroup B Strepto|"+\
	". e HydrocephalusGroup Streptoco|"+\
	".  and SepticemiaInfected Baclof|"+\
	", d Baclofen Pumpremoval c/b hem|"+\
	".  and evacuationMRSA and GNR Ve|"+\
	". iated PneumoniaSubglottic sten|"+\
	". utropenic feverCommunity acqui|"+\
	". uired pneumoniaSevere sepsis##|"+\
	". ##Severe sepsisAnemia#########|"+\
	". #########AnemiaStage II breast|"+\
	". I breast cancerGroup B Strepto|"+\
	". e HydrocephalusGroup Streptoco|"+\
	".  and SepticemiaInfected Baclof|"+\
	", d Baclofen Pumpremoval c/b hem|"+\
	".  and evacuationMRSA and GNR Ve|"+\
	". iated PneumoniaSubglottic sten"

	x_y = map(lambda line: (line[2:], line[0]), data.split('|'))
	X, Y = zip(*x_y)

	import string

	tag_lst = [',','.']
	char_lst = list(string.printable)
	char2idx = dict([(c, i) for i, c in enumerate(char_lst)] )  
	tag2idx = dict([(c, i) for i, c in enumerate(tag_lst)] )
	Y = map(lambda c: tag2idx[c], Y)
	X = map(lambda lst: [char2idx[v] for v in lst], X)
		
	VOCAB_SIZE = len(char_lst)
	EMBEDDINGS_SIZE = 10
	NUM_OF_LAYERS = 1
	STATE_SIZE = 64
	NUM_OF_CLASSES = len(tag_lst)

	specs = [VOCAB_SIZE, EMBEDDINGS_SIZE, STATE_SIZE, NUM_OF_CLASSES, NUM_OF_LAYERS, 'lstm', \
			'Adagrad', 'a.model']
	rnn = RNN(specs)

	batch_size = 5
	max_iter = 20
	display_every = 1
	rnn.train(X, Y, X, Y, batch_size, max_iter, display_every)

	y_pred = rnn.predict([X[-2]])
	print "predict ",X[-2], " : ",y_pred


			

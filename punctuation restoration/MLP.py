"""
mlp.py
	A Dynet MLP lookup plus one layer model. 
	Using it to train and predict classes(tags/chars/words) for sequences of elements(chars/words).

Usage:

import MLP

#build a new MLP, HIDDEN_SIZE = EMBEDDINGS_SIZE*feature_size(window_size)
#specs = [VOCAB_SIZE, EMBEDDINGS_SIZE, HIDDEN_SIZE, NUM_OF_CLASSES, optimizer]
specs = [100, 25, 30, 2, 'lstm', 'Adagrad']
mlp = MLP(specs)
mlp.train(batches_x, batches_y)
y_pred_lst = mlp.predict(x_lst)
mlp.save(path)

#load from path
mlp1 = MLP()
mlp1.load(path)
mlp1.train(batches_x, batches_y)
y_pred_lst = mlp1.predict(x_lst)
mlp1.save(other_path)
"""

import _dynet as dn
dyparams = dn.DynetParams()
dyparams.from_args()
dyparams.set_mem(4096)
dyparams.set_random_seed(801)
dyparams.init() # or init_from_params(dyparams)
from dynet import *

import numpy as np

class MLP:
	#create a MLP object
	def __init__(self, spec=[]):
		self.model = dn.Model()
		if spec != []:
			self.initial(spec)

	#initial a nn
	def initial(self, spec):
		[VOCAB_SIZE, EMBEDDINGS_SIZE, HIDDEN_SIZE, NUM_OF_CLASSES, NUM_OF_LAYERS, optimizer, model_path] = spec
		self.spec = spec 
		self.optimizer = optimizer
		self.model_path = model_path

		if self.optimizer == 'Adagrad':
			self.trainer = dn.AdagradTrainer(self.model)
		elif self.optimizer == 'SGD':
			self.trainer = dn.SimpleSGDTrainer(self.model)
		else:
			self.trainer = dn.AdamTrainer(self.model)
			#self.trainer = dn.AdadeltaTrainer(self.model, rho= 0.9)

		self.lookup = self.model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
		self.W = self.model.add_parameters((NUM_OF_CLASSES, HIDDEN_SIZE))
		self.B = self.model.add_parameters((NUM_OF_CLASSES))


	#calculate probs for one sample x
	def get_prob(self, x):

		lookup = self.lookup
		emb_vectors = [lookup[i] for i in x]
		net_input = dn.concatenate(emb_vectors)
		w = dn.parameter(self.W)
		b = dn.parameter(self.B)

		return dn.softmax( (w*net_input) + b)


	def train(self, X, Y, x_dev, y_dev, batch_size, max_iter, display_every=1):

		batches_x, batches_y = self.get_batch(X, Y, batch_size)
		batches_x_dev, batches_y_dev = self.get_batch(x_dev, y_dev, batch_size)
		self.train_batches(batches_x, batches_y, batches_x_dev, batches_y_dev, max_iter, display_every)


	#train a batches
	def train_batches(self, batches_x, batches_y, batches_x_dev, batches_y_dev, max_iter, display_every=1):

		best_val_acc = 0.0
		for i in xrange(max_iter):
			epoch = i+ 1
			
			for X, Y in zip(batches_x, batches_y):
				dn.renew_cg()
				losses = []
				for x, y in zip(X, Y):
					probs = self.get_prob(x)
					#print probs.npvalue()
					l = -dn.log(dn.pick(probs, y))
					losses.append(l)
				loss = dn.esum(losses)
				loss_value = loss.value()
				loss.backward()
				self.trainer.update()
		   
			train_acc = self.validate(batches_x, batches_y)
			val_acc = self.validate(batches_x_dev, batches_y_dev)

			print "\repoch:%d train_loss:%.4f train_acc: %.4f val_acc:%.4f  "%(epoch, loss_value, \
				train_acc, val_acc),
			sys.stdout.flush()

			if val_acc > best_val_acc :  #and train_acc > 0.98:
				best_val_acc = val_acc
				self.save(self.model_path)
				print "\nupdate model:",self.model_path
			elif epoch % display_every == 0:
				print	


	#predict a list [sample1, sample2, ....]
	def _predict(self, X):
		dn.renew_cg()
		pred = []
		for x in X:
			probs = self.get_prob(x)
			pred += [np.argmax(probs.npvalue())]
			
		return pred


	#predict a list [sample1, sample2, ....]
	def predict(self, x_lst, batch_size=1):

		y_pred = []
		for i in xrange(0, len(x_lst), batch_size):	
			y_pred += self._predict(x_lst[i:i+batch_size])

		return y_pred
	
	
	#validate batches
	def validate(self, testX, testY):
		acc = []
		for X, Y in zip(testX, testY):
			pred = self._predict(X)
			for p, y in zip(pred, Y):
				if p == y:
					acc += [1]
				else:
					acc += [0]

		return np.mean(acc)


	#save nn and specs
	def save(self, path):
		
		#save nn specification parameters
		spec = map(lambda x: "%d"%(x), self.spec[:-2]) + self.spec[-2:]
		with open(path +'.spec', 'w') as fd:
			fd.write('\n'.join(spec))
		
		#save the weigths
		self.model.save(path, [self.lookup, self.W, self.B])			

	#load a trained model	
	def load(self, path):

		#load the parameters 
		with open(path +'.spec', 'r') as fd:
			spec = fd.read().split('\n')
		spec = map(lambda x: int(x), spec[:-2]) +spec[-2:]
		
		#re-initial the MLP() with spec
		self.initial(spec)
		
		#load weights
		self.lookup, self.W, self.B = self.model.load(path)


	#put into minibatches
	def get_batch(self, X, Y, batch_size):   

		len1 = len(X)
		batched_x = []
		batched_y = []
		for i in xrange(0, len1, batch_size):

			batched_x += [X[i:i+batch_size]]
			batched_y += [Y[i:i+batch_size]]

		return batched_x, batched_y

	
#testing
if __name__=="__main__":

	mlp = MLP([5,10,20,3, "SGD"])

	mlp.train([[[1,2],[3,2]],[[1,0],[2,0]]], [[1,2], [0,2]], 20)
	print
	print mlp._predict([[1,0]])

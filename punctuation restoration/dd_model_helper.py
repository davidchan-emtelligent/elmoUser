"""
dd_model_help.py
	provides methods to support to create and train models, and to predict.

usage: from dd_model_helper import load_model
"""
import sys
import json
import string
import copy
import numpy as np

from RNN import RNN
from MLP import MLP

from string import maketrans
trantab = maketrans("0123456789", "dddddddddd")

from nltk.tokenize import TweetTokenizer

def Tok(sentence):

	return TweetTokenizer().tokenize(sentence)
	#return sentence.split()


#----------------------- loading model --------------------
def load_model(model_path):

	model_name = '.'.join(model_path.split('.')[:-1])
	vocab2idx_path = model_name + '_vocab2idx.json'
	parameters_path = model_name + '_parameters.json'

	with open(vocab2idx_path, 'r') as fj:
		vocab2idx = json.load(fj)
		
	with open(parameters_path, 'r') as fj:
		parameters = json.load(fj)


	#load from path
	if parameters['nn'] == 'mlp':
		clf=MLP()
	elif parameters['nn'] == 'rnn':
		clf = RNN()
	else:
		print "ERROR in load_model()!"
		sys.exit(0) 

	clf.load(model_path)

	return [clf, vocab2idx, parameters]



#----------------------get features for training and testing---------
#padding a feature
def get_feature(h_seg, t_seg, padding, head, tail):
	
	h_feats = Tok(h_seg)[-head:]
	t_feats = Tok(t_seg)[:tail]
		
	len_h = len(h_feats)
	if len_h < head:
		h_feats = [padding]*(head - len_h) + h_feats
		
	len_t = len(t_feats)
	if len_t < tail:
		t_feats += [padding]*(tail - len_t)
		
	return h_feats + t_feats


#get features at the segments at separator
def paddingAtSeparator(sent_lst, separator, tag, padding, head, tail):
				
	segment_lst = map(lambda sent: sent.split(separator), sent_lst)
	x_features = []
	y_labels = []
	words = []
	for segments in segment_lst:
		if len(segments) > 1:
			
			for i, h_seg in enumerate(segments[:-1]):

				t_seg = segments[i+1]
				if h_seg == '' or t_seg == '':
					continue
					
				h_seg = h_seg.translate(trantab)
				t_seg = t_seg.translate(trantab)
				
				x_features += [get_feature(h_seg, t_seg, padding, head, tail)]
				y_labels += [tag]
				
				words += Tok(h_seg)
			
			words += Tok(t_seg)
			
	vocab_lst = list(set(words +[padding]))
	
	return x_features, y_labels, vocab_lst


#padding features for each word ending
def padding_feature_word_wise(head_sentences, tail_sentences, tag, padding, head, tail):
			   
	x_features = []
	y_labels = []
	words = []
	n_head = head - 1 
	for (head_sent, tail_sent) in zip(head_sentences, tail_sentences):
		
			head_sent = head_sent.translate(trantab)
			tail_sent = tail_sent.translate(trantab)
			
			head_tokens = Tok(head_sent)
			tail_tokens = Tok(tail_sent)[:tail]
			
			len_h = len(head_tokens)			
			head_tokens = [padding]*(n_head) + head_tokens
			
			len_t = len(tail_tokens)
			if len_t < tail:
				tail_tokens += [padding]*(tail - len_t)
				
			tokens = head_tokens+tail_tokens

			window = head + tail
			for i in xrange(len_h):
				x_features += [tokens[i:i+window]]
				y_labels += [' ']
			y_labels[-1] = tag
			words += tokens
			
	vocab_lst = list(set(words +[padding]))
	
	return x_features, y_labels, vocab_lst


#get word based features with label comma/period
def word_features_comma_model(comma_lst, period_lst, train_ratio, padding, head, tail):
	separator = ', '
	tag = ','

	#get padded features 1: for comma labelled samples
	portion1 = int(train_ratio*len(comma_lst))
	testing1 = comma_lst[portion1:]
	training1= comma_lst[:portion1]

	x_features1, y_labels1, vocab_lst1 = \
		paddingAtSeparator(training1, separator, tag, padding, head, tail)

	x_testing1, y_testing1, vocab_lst_testing1 = \
		paddingAtSeparator(testing1, separator, tag, padding, head, tail)

	"""
	print "features 1: comma labels"
	print "data:", len(comma_lst), " train:", len(x_features1), " test:", len(x_testing1)
	print '\n'.join(comma_lst[:3])
	for (x, y) in zip(x_features1, y_labels1)[:3]:
		print x, [y]
	print len(x_features1), len(x_testing1)
	"""

	#get padded features 2: for period labelled samples for long sentences
	len2_period_lst = []
	len7_period_lst = []
	for sent in period_lst:
		sent_len = len(sent.split())
		if sent_len < 7:
			if sent_len < 3:
				len2_period_lst += [sent]
			else:
				len7_period_lst += [sent]

	len7_period_lst = period_lst
	#print len(len2_period_lst), len(len7_period_lst)

	portion2 = int(train_ratio*len(len7_period_lst))
	testing2 = len7_period_lst[portion2:]
	training2 = len7_period_lst[:portion2]

	separator = '\n'
	tag = '.'
	x_features2, y_labels2, vocab_lst2 = \
		paddingAtSeparator(['\n'.join(training2) ], separator, tag, padding, head, tail)

	x_testing2, y_testing2, vocab_lst_testing2 = \
		paddingAtSeparator(['\n'.join(testing2) ], separator, tag, padding, head, tail)
	"""
	print "\nfeatures 2: period labels for long sentences"	
	print "data:", len(len7_period_lst), " train:", len(x_features2), " test:", len(x_testing2)
	print '\n'.join(training2[:3])
	for (x, y) in zip(x_features2, y_labels2)[:3]:
		print x, [y]
	"""

	#get padded features 3: augment period labelled samples with short sentences.
	training3 = len2_period_lst

	pairs = []
	shuffle_words = copy.deepcopy(training3)
	for i in xrange(1):
		shuffle(shuffle_words)
		pairs += map(lambda (single, shuf): single + separator + shuf, zip(training3, shuffle_words))
	pairs =  pairs #+ ['COPD\nHypertension', 'Hypertension\ndiarrhea']

	x_features3, y_labels3, vocab_lst3 = \
		paddingAtSeparator(pairs, separator, tag, padding, head, tail)
	"""
	print "\nfeatures 3: augment period labels with short sentences" 
	print "pairs:", len(pairs), " train:", len(x_features3)   
	for (x, y) in zip(x_features3, y_labels3)[:3]:
		print x, [y]	
	"""

	vocab_lst = vocab_lst1 + vocab_lst2 + vocab_lst3
	vocab_testing = vocab_lst_testing1 + vocab_lst_testing2
	word_features = x_features1 + x_features2 + x_features3
	tag_labels = y_labels1 + y_labels2 + y_labels3
	word_testing = x_testing1 + x_testing2 
	tag_testing = y_testing1 + y_testing2 

	return word_features, tag_labels, word_testing, tag_testing, list(set(vocab_lst + vocab_testing))


#get word based features with label linebreak/period
def word_features_linebreak_model(period_lst, train_ratio, padding, head, tail):
	tag = '.'
	padding = '\04' #  '~'

	#get features for tarin and test data
	portion1 = int(train_ratio*len(period_lst))
	test_sentences = period_lst[portion1:]
	train_sentences = period_lst[:portion1]
	tail_sentences = period_lst   # + linebreak_lst

	"""
	print "test:", len(test_sentences), "train:", len(train_sentences), "tail:", len(tail_sentences)
	print train_sentences[:5]
	"""					  

	word_testing, tag_testing, vocab_testing = \
		padding_feature_word_wise(test_sentences[:-1], test_sentences[1:], tag, padding, head, tail) 

	#get padded features with different shuffled tail sentences
	word_features = []
	tag_labels = []
	vocab_lst = []
	for i in xrange(1):
		#print sentences[3]
		shuffle(tail_sentences)
		word_features1, tag_labels1, vocab_lst1 = \
			padding_feature_word_wise(train_sentences, tail_sentences, tag, padding, head, tail)
		word_features += word_features1
		tag_labels += tag_labels1
		vocab_lst += vocab_lst1
	vocab_lst = list(set(vocab_lst))

	return word_features, tag_labels, word_testing, tag_testing, list(set(vocab_lst + vocab_testing))


#word based features to char based features
def word_based_2_char_based(word_features, padding, head_old, tail_old, head_new, tail_new):
			   
	char_features = []
	for tokens in word_features:
		
		head_chars = ' '.join(tokens[:head_old])[-head_new:].replace(padding + ' ', padding+padding)
		tail_chars = ' '.join(tokens[-tail_old:])[:tail_new].replace(' ' + padding, padding+padding)

		len_h = len(head_chars)
		if len_h < head_new:
			head_chars = padding*(head_new - len_h) + head_chars

		len_t = len(tail_chars)
		if len_t < tail_new:
			tail_chars += padding*(tail_new - len_t)

		char_features += [head_chars + tail_chars]
	
	return char_features


from random import shuffle
#shuffle
def shuffle_features_labels(x_features, y_labels):
	
	xy_zip = zip(x_features, y_labels)
	shuffle(xy_zip)
	x_unzip, y_unzip = zip(*xy_zip)
	
	return x_unzip, y_unzip


#convert features and labels to integer representation
def features_label_2_val(x_features, y_labels, x_testing, y_testing, vocab_lst, tag_lst, padding):

	x_features, y_labels = shuffle_features_labels(x_features, y_labels)

	tag2idx = dict([(c, i) for i, c in enumerate(tag_lst )] )
	vocab2idx = dict([(v, i) for i, v in enumerate(vocab_lst)] ) 

	Y = map(lambda c: tag2idx[c], y_labels)
	X = map(lambda lst: [vocab2idx[v] for v in lst], x_features)

	Y_test = map(lambda c: tag2idx[c], y_testing)
	X_test = map(lambda lst: [vocab2idx[v] for v in lst], x_testing)

	return X, Y, X_test, Y_test, vocab2idx


#get features and labels for training a comma clf model
def get_comma_features_labels(comma_lst, period_lst, head=5, tail=3, char_based=False, nn='mlp', \
	head_new=25, tail_new=15):

	padding = '\04' #  '~'
	train_ratio = 0.9
	tag_lst=[',', '.']
	target_names = ['comma', 'period']

	word_features, tag_labels, word_testing, tag_testing, word_lst = \
		word_features_comma_model(comma_lst, period_lst, train_ratio, padding, head, tail) 
	word_features, tag_labels = shuffle_features_labels(word_features, tag_labels)  
	
	if char_based:
		window_size = head_new + tail_new
		EMBEDDINGS_SIZE = 128
		vocab_lst = list(set(string.printable+padding))

		x_features = word_based_2_char_based(word_features, padding, head, tail, head_new, tail_new)
		y_labels = tag_labels
		x_testing = word_based_2_char_based(word_testing, padding, head, tail, head_new, tail_new)
		y_testing = tag_testing
		
	else:
		window_size = head + tail
		EMBEDDINGS_SIZE = 300
		vocab_lst = list(set(word_lst + ['unk0',padding])) 

		x_features = word_features
		y_labels = tag_labels
		x_testing = word_testing
		y_testing = tag_testing
		
	"""  
	print "\nfeatures: " 
	print "train:", len(x_features)  , "vocab:", len(word_lst)  
	for (x, y) in zip(x_features, tag_labels)[:3]:
		print [x], [y]
	#""" 
		
	#convert features and labels to digits
	X, Y, X_test, Y_test, vocab2idx = \
		features_label_2_val(x_features, y_labels, x_testing, y_testing, vocab_lst, tag_lst, padding)

	parameters = {'char_based': char_based, 'head': head, 'tail': tail, 'head_char': head_new, \
		'tail_char': tail_new, 'padding': padding, 'embedding_size': EMBEDDINGS_SIZE, \
		'tag_lst': tag_lst, 'nn': nn} 
	
	return  X, Y, X_test, Y_test, vocab2idx, tag_lst, parameters, window_size, target_names


#get features and labels for training a linebreak clf model
def get_linebreak_features_labels(period_lst, head=5, tail=3, char_based=False, nn='mlp', \
	head_new=25, tail_new=15):

	tag = '.'
	padding = '\04' #  '~'
	train_ratio = 0.9
	tag_lst = [' ', '.']
	target_names = ['space', 'period']

	word_features, tag_labels, word_testing, tag_testing, word_lst = \
				word_features_linebreak_model(period_lst, train_ratio, padding, head, tail)

	word_features, tag_labels = shuffle_features_labels(word_features, tag_labels)
	
	if char_based:
		window_size = head_new + tail_new
		EMBEDDINGS_SIZE = 128
		vocab_lst = list(set(string.printable+padding))

		x_features = word_based_2_char_based(word_features, padding, head, tail, head_new, tail_new)
		y_labels = tag_labels
		x_testing = word_based_2_char_based(word_testing, padding, head, tail, head_new, tail_new)
		y_testing = tag_testing

	else:
		window_size = head + tail
		EMBEDDINGS_SIZE = 300
		vocab_lst = list(set(word_lst + ['unk0',padding])) 

		x_features = word_features
		y_labels = tag_labels
		x_testing = word_testing
		y_testing = tag_testing

	"""  
	print "\nfeatures: " 
	print "train:", len(x_features)  , "vocab:", len(word_lst)  
	for (x, y) in zip(x_features, tag_labels)[:3]:
		print [x], [y]
	#"""
	
	#convert features and labels to digits
	X, Y, X_test, Y_test, vocab2idx = \
		features_label_2_val(x_features, y_labels, x_testing, y_testing, vocab_lst, tag_lst, padding)

	parameters = {'char_based': char_based, 'head': head, 'tail': tail, 'head_char': head_new, \
		'tail_char': tail_new, 'padding': padding, 'embedding_size': EMBEDDINGS_SIZE, \
		'tag_lst': tag_lst, 'nn': nn} 

	return  X, Y, X_test, Y_test, vocab2idx, tag_lst, parameters, window_size, target_names


#----------------------prediction classifier: space/period---------for linebreak tag
#classify a list of features to given labels
def linebreak_classifier(pairs, separator, clf):	
	[model, vocab2idx, parameters] = clf

	head = parameters['head']
	tail = parameters['tail']
	padding = parameters['padding']
	
	word_testing, y_testing, vocab_testing = \
		paddingAtSeparator(pairs, separator, 'anything', padding, head, tail)
		
	X_test = []		
	if parameters['char_based']:
		head_char = parameters['head_char']
		tail_char = parameters['tail_char']
		x_testing = word_based_2_char_based(word_testing, padding, head, tail, head_char, tail_char)
		
		X_test = map(lambda lst: [vocab2idx[v] for v in lst], x_testing)
		
	else:
		x_testing = word_testing
		
		for lst in x_testing:
			try:
				X_test += [[vocab2idx[v] for v in lst]]
			except:
				for i in xrange(len(lst)):
					if lst[i] not in vocab2idx:
						#print [lst[i]],'---------------------'
						lst[i] = 'unk0'
				X_test += [[vocab2idx[v] for v in lst]]		
			
		
	y_pred = model.predict(X_test)
	#for (x, y) in zip(x_testing, y_pred):
	#	print x, [y]
	
	tag_lst = parameters['tag_lst'] 
	
	return map(lambda p: tag_lst[p], y_pred)
	

#restore linebreak				  
def restore_linebreak_by_clf(notes, notes_tags, notes_bad, notes_seg_idxs, clf):

	new_notes = []
	tags = []
	idxs = []
	pairs = []
	for i, (note, t, b, idx_lst) in enumerate(zip(notes[:-1], notes_tags[:-1], notes_bad[:-1],\
		 notes_seg_idxs[:-1])):

		if t == '_' or t == '.':
			new_notes += [note]
			tags += [t]
			idxs += [idx_lst]
			"""
		elif b == 0:
			new_notes += [note]
			tags += ['.']
			idxs += [idx_lst]
			#"""			
		elif t == '\n': 
			new_notes += [note]
			tags += [t]
			idxs += [idx_lst]
			pairs += [note + '\n' + notes[i+1]]
			
	new_notes += [notes[-1]]
	tags += ['.']
	idxs += [notes_seg_idxs[-1]] 
	
	y_pred = linebreak_classifier(pairs, '\n', clf)
	
	j = 0
	for i in xrange(len(new_notes)):
		if tags[i] == '\n':
			tags[i] = y_pred[j]
			j += 1

	#overright the predict to ' ' if the last char is ','
	if new_notes[i][-1] == ',':
		tags[i] = ' '

	return new_notes, tags, idxs


#----------------------prediction classifier: comma/period---------for comma tag
#classify a list of sentence with comma
def comma_classifier(sentence, clf):
	[model, vocab2idx, parameters] = clf
	
	head = parameters['head']
	tail = parameters['tail']
	padding = parameters['padding'] 
	
	word_testing, y_labels, _ = \
		paddingAtSeparator([sentence], ', ',  ',', padding, head, tail)
	
	X_test = []
	if parameters['char_based']:
		head_char = parameters['head_char']
		tail_char = parameters['tail_char']
		x_testing = word_based_2_char_based(word_testing, padding, head, tail, head_char, tail_char)
		
		X_test = map(lambda lst: [vocab2idx[v] for v in lst], x_testing)  
		
	else:
		x_testing = word_testing  
		
		for lst in x_testing:
			try:
				X_test += [[vocab2idx[v] for v in lst]]
				
			except:
				for i in xrange(len(lst)):
					if lst[i] not in vocab2idx:
						#print [lst[i]],'---------------------'
						lst[i] = 'unk0'
				X_test += [[vocab2idx[v] for v in lst]]		
		
		
	y_pred = model.predict(X_test)
	
	tag_lst = parameters['tag_lst']
	
	return map(lambda y: tag_lst[y], y_pred)

#restore comma 
def restore_comma_by_clf(notes, notes_tags, notes_seg_idxs, clf):
	new_notes = []
	tags = []
	idxs = []
	for (note, t, idx_lst) in zip(notes, notes_tags, notes_seg_idxs):

		if t == '_':
			new_notes += [note]
			tags += [t]
			idxs += [idx_lst]
			
		elif len(note.split(', ')) > 1:
			y_pred = comma_classifier(note, clf)
			lst = note.split(', ')
			for (x, y, idx) in zip(lst[:-1], y_pred, idx_lst):
				new_notes += [x]
				tags += [y]
				idxs += [[idx]]
 
			new_notes += [lst[-1]]
			tags += ['.']
			idxs += [[idx_lst[-1]]]
			
		elif t in ['\n', '.', ' ']: 
			new_notes += [note]
			tags += [t]
			idxs += [idx_lst]
			
	tags[-1] = '.'

	return new_notes, tags, idxs


from sklearn.metrics import classification_report
#get score display string
def get_scores(truth, predict, target_names = ['comma', 'period']):
	
	n = len(target_names)
	mat = np.zeros((n,n), dtype=int)
	accur = 0
	for (t, p) in zip(truth, predict):
		mat[p, t] += 1
		if t == p:
			accur += 1
	accur = float(accur)/np.sum(mat)
		
	ret = classification_report(truth, predict, target_names=target_names)+\
		'\nconfusion matrix:\n %s\n%s\n\nAccuracy: %.2f%%'%( ' '.join(target_names), \
		str(mat), accur*100 )
	
	return ret


#run testing
if __name__=="__main__":

	#1) get data-----------------
	import json
	with open ('dd_reports.json') as fj:
		dd_lst = json.load(fj)
		
	from dd_helper import *
		
	print "loading data:",len(dd_lst)
	import sys, string


	dd_notes_lst = []
	for i, d in enumerate(dd_lst[:1000]): #[1165:1166]):   #[57:58]):   #[19574:19576]:
		idx = d['report_id']		
		page = d['discharge_diagnosis'].encode('utf-8')
		base = d['span_base'] 
		original_idx = d['original_idx']

		dd = parse2segments(page)		
		notes = segments2notes(dd, i, bad_len=48)		
		dd_type = type_classifier(notes)
		
		#print dd	
		dd.update({'dd_idx': i})
		dd.update({'dd_type': dd_type})

		dd.update({'report_id': idx})
		dd.update({'original_idx': original_idx})
		dd.update({'span_base': base})
		dd.update({'original_idx': original_idx})
		dd.update({'original_file': d['original_file']})

		dd.update({'info': notes['info'], 'notes_bad': notes['notes_bad'],'notes': notes['notes'],\
		'notes_tags': notes['notes_tags'], 'notes_seg_idxs': notes['notes_seg_idxs']})
				
		dd_notes_lst += [dd]

	print json.dumps(dd_notes_lst[520], sort_keys=True,indent=2)
	#----------------------------classifier: comma/period---------for comma tag
	comma_clf = load_model('model/comma_clf.model')

	idx_lst = [520,752,766,820]
	for idx in idx_lst:

		dd = dd_notes_lst[idx]		
		new_notes, tags, idxs = restore_comma_by_clf(dd['notes'], dd['notes_tags'], \
		dd['notes_seg_idxs'], comma_clf)	   
		#new_notes, tags, idxs = comma2period(dd['notes'], dd['notes_tags'], dd['notes_seg_idxs'])
		sentences, tags, spans = notes_2sentences(new_notes, tags, idxs, dd['segment_spans'])

		print '\n================\nreport_id:%s\n================(%d, %s)'%\
		(dd['report_id'],dd['dd_idx'], dd['dd_type'])
		print '------------input-------------\n', dd_lst[dd['dd_idx']]['discharge_diagnosis']
		print '------------output------------\n', sentence2string(sentences, tags)
		print spans

	#----------------------------classifier: space/period---------for linebreak tag
	linebreak_clf = load_model('model/linebreak_clf.model')

	idx_lst = [142,202,433,566,705] 
	for idx in idx_lst:
		dd = dd_notes_lst[idx]

		new_notes, tags, idxs = restore_linebreak_by_clf(dd['notes'], dd['notes_tags'], \
		dd['notes_bad'], dd['notes_seg_idxs'], linebreak_clf)		   
		new_notes, tags, idxs = restore_comma_by_clf(new_notes, tags, idxs, comma_clf)	   
		sentences, tags, spans = notes_2sentences(new_notes, tags, idxs, dd['segment_spans']) 

		print '\n================\nreport_id:%s\n================(%d, %s)'%\
		(dd['report_id'],dd['dd_idx'], dd['dd_type'])
		print '------------input-------------\n', dd_lst[dd['dd_idx']]['discharge_diagnosis']
		print '------------output------------\n', sentence2string(sentences, tags)
		print spans 



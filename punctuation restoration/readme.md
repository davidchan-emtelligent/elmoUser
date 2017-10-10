## Problem

The punctuation ambiguities in a discharge diagnosis summary (medical note) mainly are colon, line-break, comma and period. Our goal is to examine all these punctuations and restore the incorrect punctuations in summary. Finally, we can provide a list of correct bullet points.

<br> 

## Method

We suggest to employ a hybrid method combing a rule-based classifer with ML classifers to classify the type of problem and restore the incorrect punctuations. The method is:

1) Split the raw text into fragments and record the existing punctuations at their two ends.

2) Using a rule-based classifier to remove the existing bullet points and assign a tag (punctuation) to each fragment. These can keep the fragments which are well formatted.

3) Collect the statistic of punctuations for the whole summary: # of bullets, # of periods, # of line-breaks, # of lines and # of heading (fragment with a colon is a possible heading).

4) Based on the statistic, we can decide if the summary is well formated or not. And we also can find out the ambiguities and therefor the type of problem. Furthermore we can find out which fragment has been formated well or not.

5) According to the type of problem in each summary, we can choose either rule-based or ML method to restore the punctuations for the not well formated fragments.

6) Re-construct the list of bullet points with desire format.

<br> 

## Rules

##### Rules to assign tags:
	
tags: '#' for bullet point, '_' for heading.

	prev_tag, end_tag ==> note tag
	' ', '\n'  ==>  '\n'
	' ', '. '  ==>  '. '
	' ', ') '  ==>  '#'	(check bullet?)
	' ', ','  ==>  ','
	'\n', ','  ==>  ','
	'\n', '.'  ==>  '.'
	'\n', '\n'  ==>  '\n'	
	'\n', ': '  ==>  '_'	(check heading_list)
	'\n', ':\n'  ==>  '_'
	and first char '-' or '#' ==> '#'


##### Rules to restore punctuations:

	- tag '_': keep.
	- tag '#': remove.
	- tag '.': keep.
	- tag ',' or '\n': check and restore.


##### Rules to restoration for types of problem (decision tree in order):

	- 'good': if n_bullet == n_period or (n_bullet == n_bline and n_period == 0):
		Do nothing.

	- 'bullet_1': if n_bullet > 0:
		Use restore_bullet_2sentences()

	- 'period_1_bullet_0': if n_period > 0 and n_bullet == 0:
		Use restore_period_2sentences()

	- 'comma_0': one line match one heading and not comma:
		One valid bullet point. Do nothing.

	- 'comma_1': one line match one heading and n_comma > 0:
		Use restore_comma_2sentences() or simply replace ',' to '.'.

	- 'period_0_bullet_0_short': n_period == 0 and n_bullet == 0, and len(fragment) < bad_length(45-50):
		Use restore_linebread_2sentences() to restore '\n' or simply replace '\n' to '.'.
		Use restore_comma_2sentences() to restore ',' as well.

	- 'period_0_bullet_0_long': n_period == 0 and n_bullet == 0, and len(fragment) >= bad_length(45-50):
		Use restore_linebread_2sentences() to restore '\n'.
		Use restore_comma_2sentences() to restore ',' as well.

<br> 

## Hybrid Classifier

The hybrid classifier includes:

##### 1. restore_punctuation.py

Restore punctuations in the discharge diagnosis of a list of medical reports.

	usage: restore_punctuation.py -i csv_path -o output_path -m model_path
		-i : a csv file has a list of medical reports.
		-o : a json file has a list of dicts.
		     Each dict has keys: 'report_id', 'sentences', 'tags', 'sentences_spans'.
		-m : ML classifier model.
		     Default is to create a new model.

##### 2. dd_helper.py

methods:

	parse2segments(): parse a raw summary to segments with punctuations at 2 ends.
	segments2notes(): remove existing bullet points, assign tags for each segment and get the punctuation statistic.
	type_classifier(): classify the type of problem for a summary.
	restore_bullet_2sentences(): restore punctuations for a summary with bullet points.
	restore_period_2sentences(): restore punctuations for a summary with periods.

	usage:  from dd_helper import *

##### 3. dd_model_helper.py

methods:

	padding_feature_word_wise(): pad 2 segments (head and tail) to a sample feature in words.
	word_based_2_char_based(): convert word_based feature to char_based feature.
	restore_linebreak_2sentences(): restore linebreaks with ML classifer.
	restore_comma_2sentences(): restore commas with ML classifer.

	usage:  from dd_model_helper import *

##### 4. MLP.py

Dynet MLP nueral network consists of train(), predict(), save() and load().

	usage:  from MLP import MLP
		mlp_model = MLP(spec)

##### 5. RNN.py

Dynet RNN nueral network consists of train(), predict(), save() and load().

	usage:  from RNN import RNN
		mlp_model = MLP(spec)

<br> 

## Test case

##### 1. Data:

We use the data set 'all_rad_output.csv'. It has 203180 reports in which 99847 reports has non-empty discharge diagnoses(dd).

##### 2. Restore punctuations for different types of problem:

	'good': 35848

	'bullet_1': 12048
	
		================
		report_id:97981
		================(70, bullet_eq_period)
		
		------------**input**-------------
		
		1.  Status post minimally invasive mitral valve repair via a

		right thoracotomy.

		2.  Hypothyroidism.
		
		------------**output**------------
		
		1. Status post minimally invasive mitral valve repair via a right thoracotomy.
		2. Hypothyroidism.

	'period_1_bullet_0': 6119
	
		================
		report_id:97959
		================(49, period_1_bullet_0)
		
		------------**input**-------------
		
		Gastric perforation from marginal

		ulcer.
		
		------------**output**------------
		
		1. Gastric perforation from marginal ulcer.

	'comma_1': 2205
	
		================
		report_id:97965
		================(55, comma_1)
		
		------------**input**-------------
		
		etoh intoxication, etoh withdrawal seizure
		
		------------**output**------------
		
		1. etoh intoxication.
		2. etoh withdrawal seizure.

	'period_0_bullet_0_short': 29207
	
		================
		report_id:97910
		================(0, period_0_bullet_0_short)
		
		------------**input**-------------
		
		Listeria meningitis

		Ulcerative colitis
		
		------------**output**------------
		
		1. Listeria meningitis.
		2. Ulcerative colitis.


	'period_0_bullet_0_long': 14420
	
		================
		report_id:98176
		================(247, period_0_bullet_0_long)
		
		------------input-------------
		
		Primary: Cellulitis

		Secondary: Atrial fibrillation, Chronic systolic heart failure,

		End stage renal disease on [**Name (NI) 2252**]
		
		------------output------------
		
		# Primary
		1. Cellulitis.
		# Secondary
		2. Atrial fibrillation.
		3. Chronic systolic heart failure.
		4. End stage renal disease on [**Name (NI) 2252**].

 
##### 3. ML classifier: comma_clf.model

 - Data samples are collected from 'good' sentences and other sentences with tag '.'.

 - For label comma ',', we collect 2920 identical sentences (from 7886). They give us 3189 samples.

 - For label period '.', we collect 17380 identical sentences (from 86861). After augmentation with shuffled tail sentences, we got 22118 samples.

 - Shuffle and split them into 23254 training and 2053 testing samples.

 - Using head = 6 and tail = 3 words, we create 9 words features for our nueral network model.

 - While best char-based (head = 25 and tail = 15 chars) LSTM model gives the best accuracy 0.9459, word-based MLP model gives a better accuracy 0.9489:

		vocabs: 8712  tags: 2  len(feature_vec): 9
		[8712, 300, 2700, 2, 1, 'Adam', 'model/comma_clf.model']
		epoch:2 train_loss:0.5626 train_acc: 0.9831 val_acc:0.9464    
		update model: model/comma_clf.model
		epoch:3 train_loss:0.3029 train_acc: 0.9917 val_acc:0.9469   
		update model: model/comma_clf.model
		epoch:5 train_loss:0.0266 train_acc: 0.9972 val_acc:0.9479    
		update model: model/comma_clf.model
		epoch:10 train_loss:0.0025 train_acc: 0.9986 val_acc:0.9474     
		epoch:12 train_loss:0.0020 train_acc: 0.9986 val_acc:0.9484    
		update model: model/comma_clf.model
		epoch:15 train_loss:0.0008 train_acc: 0.9990 val_acc:0.9484    
		epoch:16 train_loss:0.0008 train_acc: 0.9986 val_acc:0.9489   
		update model: model/comma_clf.model
		epoch:20 train_loss:0.0005 train_acc: 0.9990 val_acc:0.9464     
		time: 32.4604980946

			     precision    recall  f1-score   support

		      comma       0.87      0.79      0.83       316
		     period       0.96      0.98      0.97      1737

		avg / total       0.95      0.95      0.95      2053

		confusion matrix:
		 comma period
		[[ 249   38]
		 [  67 1699]]

		Accuracy: 94.89%

##### 4. ML classifier: linebreak_clf.model

- We collect data samples from 'good' sentences and other sentences with tag '.'.

- Samples are collected word-wise through 2 concatenated sentences (head and tail sentences). After augmentation with shuffled tail sentences, we got 66735 training sample and 7546 testing sample.

- Using head = 6 and tail = 3 words, we create 9 words features for our nueral network model.

- Word-based MLP model gives a better accuracy 0.9829:

		vocabs: 8021  tags: 2  len(feature_vec): 9
		[8021, 300, 2700, 2, 1, 'Adam', 'model/linebreak_clf.model']
		epoch:1 train_loss:5.5361 train_acc: 0.9880 val_acc:0.9780   
		update model: model/linebreak_clf.model
		epoch:2 train_loss:3.5135 train_acc: 0.9962 val_acc:0.9823   
		update model: model/linebreak_clf.model
		epoch:3 train_loss:0.3759 train_acc: 0.9984 val_acc:0.9826   
		update model: model/linebreak_clf.model
		epoch:4 train_loss:0.0149 train_acc: 0.9987 val_acc:0.9829   
		update model: model/linebreak_clf.model
		epoch:5 train_loss:0.0048 train_acc: 0.9988 val_acc:0.9825  
		epoch:10 train_loss:0.0045 train_acc: 0.9996 val_acc:0.9817     
		epoch:15 train_loss:0.0001 train_acc: 0.9996 val_acc:0.9793      
		epoch:20 train_loss:0.0000 train_acc: 0.9996 val_acc:0.9786      
		time: 77.5569369793

			     precision    recall  f1-score   support

		      space       0.99      0.98      0.99      5793
		     period       0.95      0.98      0.96      1737

		avg / total       0.98      0.98      0.98      7530

		confusion matrix:
		 space period
		[[5697   33]
		 [  96 1704]]

		Accuracy: 98.29%

##### 5. Model accuracy
	--------------------------------------------------
	|                 | comma_clf  |  linebreak_clf  |  
	-------------------------------------------------- 
	| word-based MLP  |  0.9489    |     0.9829      |  
	| char-based LSTM |  0.9459    |     0.9685      |  
	--------------------------------------------------


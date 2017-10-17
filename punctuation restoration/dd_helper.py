# -*- coding: utf-8 -*- 
"""
dd_helper.py
	Methods to support restore_punctuations. Including the rule-based methods.

usage: from dd_helper import *
"""
import numpy as np
import sys
import string
import json
import csv
import os

if not os.path.isfile('model/heading_lower_lst.txt'):
	heading_lower_lst = ['primary diagnosis', 'secondary diagnosis',\
		'primary diagnoses', 'secondary diagnoses', \
		'primary diagonsis', 'secondary diagonsis']
else:
	with open('model/heading_lower_lst.txt', 'r') as fd:
		heading_lower_lst = fd.read().split('\n')

#---------------------- extract discharge diagnosis data from csv -------------
title_lst = ['Discharge Diagnosis:', 'Discharge Diagonsis:', 'Discharge Diagnoses:', \
		'DISCHARGE DIAGNOSIS:', 'DISCHARGE DIAGNOSES:']
def get_discharge_diagnosis(filtered_lst, title_lst, file_name):

	dd_lst = []
	no_dd_idx = []
	empty_dd_idx = []
	nnn = '\n\n\n'
	for i, (index, report) in enumerate(filtered_lst):

		raw = report['original_report']
		for title in title_lst:
			lst = []
			lst = extractList_start_end(raw, title, nnn)
			if lst == []:
				continue
			text = lst[0]
			if text.encode('utf-8').translate(None, string.punctuation).strip() != '':   
				base = len(raw.split(title)[0])+len(title)
				dd_lst += [{'report_id': report['report_id'], 'dd_title': title, \
					'discharge_diagnosis': text, 'original_file_span_base': base, \
					'original_file_idx': index, 'original_file': file_name} ]
			else:
				empty_dd_idx += [i]
			break

		if lst == []:
			no_dd_idx += [index]
	#print len(dd_lst), len(empty_dd_idx), len(no_dd_idx)

	final_empty_dd_idx = []						  
	nnn = '\n\n\n\n\n\n'
	for (index, report) in [filtered_lst[i] for i in empty_dd_idx]:

		raw = report['original_report']

		lst = []
		for title in title_lst:
			lst = extractList_start_end(raw, title, nnn)
			if lst != []:

				text = lst[0]
				if text.encode('utf-8').translate(None, string.punctuation).strip() != '' :
					base = len(raw.split(title)[0])+len(title)
					dd_lst += [{'report_id': report['report_id'], 'dd_title': title, \
						'discharge_diagnosis': text, 'original_file_span_base': base, \
						'original_file_idx': index, 'original_file': file_name} ]
				else:
					final_empty_dd_idx += [index]
				break

		if lst == []:
			no_dd_idx += [i]
			
	return dd_lst, final_empty_dd_idx, list(set(no_dd_idx))


#extract discharge diagnosis from csv file
def extract_dd_raw(input_csv_path):

	#using csvreader read raw csv file and save raw reports to tmp.csv
	output_path = 'tmp.csv'
	print "python csvreader.py -i '%s' -o '%s'"%(input_csv_path, output_path)
	os.system("python csvreader.py -i '%s' -o '%s'"%(input_csv_path, output_path) )
		
	#read tmp.csv, extract dd and save to dd_lst
	reader = csv.DictReader(open(output_path, 'r'))
	reports_lst = []
	for row in reader:
		reports_lst += [row]
	os.system("rm '%s'"%(output_path) )

	index_report_lst = [(i, r) for i, r in enumerate(reports_lst)]
	filtered_lst = filter(lambda (i, r): r['category'] == 'Discharge summary', index_report_lst)

	file_name = input_csv_path.split('/')[-1]
	dd_lst, empty_dd_idx, no_dd_idx = get_discharge_diagnosis(filtered_lst, title_lst, file_name)
	#print "num of dd:", len(dd_lst) #, len(empty_dd_idx), len(no_dd_idx)

	return dd_lst, reports_lst

#---------------------- get data ----------------------------------------------
#get sentences data with comma and with period at end
def get_data(notes, tags, bad, n_comma, valid_lst=['.']):	
	period_lst = []
	comma_lst = []
	linebreak_lst = []
	n_data = 0
	n_sample = 0
	n_non_sample = 0
	for (n, t, b) in zip(notes, tags, bad):
		
		if t in valid_lst:
			n_sample += 1
		else:
			if t == '\n':
				n_non_sample += 1	
			continue

	#filter out too long sentences
		if b > 0:
			continue
			
		n_data += 1
		if n_comma > 0:
			lst = n.split(',')
			if len(lst) > 1:
				comma_lst += [n]
				continue
				
		if t == '.':
			period_lst += [n]
			
		elif t == '\n':
			linebreak_lst += [n]
			
			
	return period_lst, comma_lst, linebreak_lst, n_data, n_sample, n_non_sample


#get sentences data to support create models: period_sentences, comma_sentences and heading_lower
def get_sentences_data(dd_lst):
	heading_lst = []

	period_lst_good = []
	comma_lst_good = []
	linebreak_lst_good = []
	period_lst_bad = []
	comma_lst_bad = []
	linebreak_lst_bad = []

	n_data = 0
	n_sample = 0
	n_non_sample = 0
	for i, d in enumerate(dd_lst[:]):
		idx = d['report_id']
		page = d['discharge_diagnosis'].encode('utf-8')
  
		dd = parse2segments(page)
		notes = segments2notes(dd, i, bad_len=48)
		dd_type = type_classifier(notes)

		n_comma = dd['info']['n_comma']
		n_heading = dd['info']['n_heading']

		if n_heading > 0:
			for (n, t) in zip(dd['notes'], dd['notes_tags']):
				if t == '_':
					len1 = len(n.split())
					if len1 > 0 and len1 < 4 :
						heading_lst += [n]

		#collect period_lst, comma_lst, linebreak_lst
		if dd_type == 'good' or dd_type == 'comma_0':
			p_lst, c_lst, l_lst, n1, n2, n3 = get_data(dd['notes'], dd['notes_tags'], \
				dd['notes_bad'],n_comma, valid_lst=['.', '\n'])
			period_lst_good += p_lst
			comma_lst_good += c_lst
			linebreak_lst_good += l_lst

		#get more data: except 1) '\n' with len() < bad_len and 2) not comma_1
		elif not dd_type == 'comma_1':
			p_lst, c_lst, l_lst, n1, n2, n3 = get_data(dd['notes'], dd['notes_tags'], \
				dd['notes_bad'],n_comma, valid_lst=['.'])
			period_lst_bad += p_lst
			comma_lst_bad += c_lst
			linebreak_lst_bad += l_lst

		n_data += n1
		n_sample += n2
		n_non_sample += n3
		
	#print "done!"
	#print len(dd_lst)
	#print len(period_lst_good), len(comma_lst_good), len(linebreak_lst_good), len(heading_lst)
	#print len(period_lst_bad), len(comma_lst_bad), len(linebreak_lst_bad)
	#print n_data, n_sample, n_non_sample

	period_lst = period_lst_good + linebreak_lst_good
	comma_lst = comma_lst_good + comma_lst_bad
	heading_lst = list(set(heading_lst))
	heading_lower_lst = list(set(map(lambda s: s.lower().translate(None, string.punctuation + \
		string.digits).strip(), heading_lst)))
	heading_lower_lst = filter(lambda s: s != '', heading_lower_lst)
		
	return list(set(period_lst)), list(set(comma_lst)), heading_lower_lst


#----------------------raw text to segments, notes, sentences ---------------------------
#extract from head to tail inclusive
def extractList_start_end(text, head, tail, inclusive = False):
	lst = text.split(head)
	ret = []
	if len(lst) > 1:
		for l in lst[1:]:
			l_split = l.split(tail)
			if len(l_split) > 1:
				if inclusive:
					out = head + l_split[0] + tail
				else:
					out = l_split[0]
				ret += [out]
			else:

				#print "extractList_start_end(): can't find tail: '%s' in %s\n%s"%(tail,l_split, url)
				continue
	return ret   #a list


#search the first non-empty char to support parse2segments
def get_start_end(text):
	i = 0
	if text == '':
		return [i, i]
	if text == ' ':
		return [i, i+1]

	while text[i] == ' ':
		i += 1
	return [i, i+ len(text.strip())]

	
#check if it is a heading sentence
def is_heading(seg):
	heading = seg.lower().translate(None, string.punctuation+string.digits).strip()
	if heading in heading_lower_lst:
		return True		
	return False


#check if not a line break (not been used)
def isnot_linebreak(note_prev, note):

	if note_prev == '' or note == '':
		return False

	if note[0].islower():
		return True

	return False


delimiter = '\04'
prev_tags = [' ', '\n']
separators = [',', ', ', ',\n', '. ', '.\n', ') ', ': ', ':\n', '\n']

separator2len = dict([(s, len(s)) for s in separators])


#parse a dd into segments
def parse2segments(text):

	if text[-1] == ',':
		text = text[:-1] + '.\n'
	else:
		text = text + '\n'

	min_ptr = 0
	max_ptr = len(text)

	#1) split into segments
	segs_str = text
	for s in separators:
		segs_str = segs_str.replace(s, delimiter)

	segs = segs_str.split(delimiter)

	#2) find spans and tags
	ptr = min_ptr	
	spans = []
	tags = []
	for i, seg in enumerate(segs):

		#get prev_tag and end_tag
		if ptr == min_ptr:
			prev_tag = '\n'
		else:
			prev_tag = text[ptr-1]

		ptr_end = ptr + len(seg)
		if ptr_end >= max_ptr:
			end_tag = '\n'
		else:
			if text[ptr_end] == '\n':
				end_tag = '\n'
			elif text[ptr_end] == ',':
				end_tag = ','
			else:
				end_tag = text[ptr_end:ptr_end+2]

		#remove double spacing
		[s, e] = get_start_end(seg)
		start = ptr + s
		end = start + e - s

		seg = text[start:end]

		#empty
		if seg == '':
			ptr += separator2len[end_tag]
			continue

		if i == 0:
			#empty, has no alphabets or digit
			if len(seg.translate(None, string.punctuation).strip()) == 0:
				ptr += len(seg)+ separator2len[end_tag]
				continue

		#not empty
		#print text[s:e], [end_tag]
		ptr = ptr_end + separator2len[end_tag]

		spans += [(start, end)]
		tags += [(prev_tag, end_tag)]

	#3) return a dict
	dd = {}
	dd['raw'] = text[:-1]
	dd['segment_spans'] = spans
	dd['segment_tags'] = tags

	return dd


#parse segments to notes with tags
def segments2notes(dd, idx, bad_len=50):
	"""
	input: a page(string)
	output: a dict with keys: raw, segment_spans, segment_tags, notes, notes_tags, 
	   notes_pos and bullets 

	Rule of notes_tags:
	prev_tag, end_tag ==> note tag
	' ', '\n'  ==>  '\n'
	' ', '. '  ==>  '. '
	' ', ') '  ==>  '#'	(check bullet?)
	'\n', '.'  ==>  '.'
	'\n', '\n'  ==>  '\n'	(check bullet?)
	'\n', ': '  ==>  '_'
	'\n', ':\n'  ==>  '_'
	bullet ==> '#'
	"""

	text = dd['raw'] 
	max_ptr = len(text)

	notes = []
	tags = []
	notes_seg_idxs = []

	n_bullet = 0
	n_hyphen_bullet = 0

	n_concate = 0
	n_not_bullet_concate = 0
	n_lb_concate = 0

	#1) create notes and tags by rules
	for i, ((s, e), (prev_tag, end_tag)) in enumerate(zip(dd['segment_spans'], dd['segment_tags'])):

		tag =''
		seg_idxs = [i]
		if prev_tag in [',', ' '] and end_tag == '\n':
			tag = '\n'

		elif prev_tag == '\n' and end_tag == '\n':
			seg = text[s:e]
			# check for 'primary,secondary,...etc.
			if is_heading(seg):
				tag = '_'
			else:
				tag = '\n'

		elif prev_tag in [',', ' '] and end_tag == ':\n':
			tag = '_'

		elif prev_tag in [',', ' '] and end_tag == ': ':
			#concate the note_prev
			tag = end_tag
			n_concate += 1

		elif prev_tag == '\n' and end_tag == ': ':
			seg = text[s:e]
			# check for 'primary,secondary,...etc.
			if is_heading(seg):
				tag = '_'
			else:
				#concate the note_prev
				tag = end_tag
				n_concate += 1

		elif prev_tag == '\n' and end_tag == ':\n':
			tag = '_'			
			
		elif prev_tag in [',', ' ', '\n'] and end_tag in [',', ', ', ',\n']:
			tag = ','
			
		elif prev_tag in [',', ' ', '\n'] and end_tag == '.\n':
			tag = '.' 

		#elif prev_tag == ' ' and end_tag in ['. ', ') ']:
		#	tag = end_tag 
		#	n_concate += 1

		#bullet point with 1,2,3,...		
		elif prev_tag in [',', ' ', '\n'] and end_tag in ['. ', ') ']:		   
			seg = text[s:e]
			words = seg.split()
			num = words[0].translate(None, string.punctuation)
			#if len(words) == 1 and (num.isdigit() or len(num) == 1):
			if len(words) == 1 and num.isdigit():
				tag = '#'
				n_bullet += 1
			else:
				tag = end_tag
				n_not_bullet_concate += 1

		if tag == '':
			print "Unvalid tags:", idx, [prev_tag, end_tag],i,text[s:e],text
			sys.exit(0)

		else:
			note = text[s:e].strip()

			#empty seg
			if note == '':
				continue

			#check error line break, eg. ...heart\nfailure.
			if len(notes) > 0:
				#bullet is not concatenated to prev segment
				if tags[-1] == ',' and tag == '#':
					tags[-1] = '.'

				#concatenate to prev segment: last tag is ': ', ') ' or ','
				elif tags[-1] in ['. ', ': ', ') ', ',']:  #not in ['_', '.', '\n','#']:
					last_tag = tags[-1]
					if last_tag == ',':
						last_tag = ', '
					#elif last_tag == ': ':
					#	print notes[-1], [last_tag], note, idx
					note_prev = notes[-1]						
					notes[-1] = note_prev + last_tag + note
					tags[-1] = tag
					notes_seg_idxs[-1] += seg_idxs					
					continue					
			
			#remove '-' and '- ' at front
			n = note[0]			
			if n == '-' or n == '#' or n == '*':			
				notes += [n]
				tags += ['#']
				notes_seg_idxs += [[]]

				len1 = len(note)
				len2 = len(note.replace(n, ' ').strip())
				note = note[len1-len2:]			
				n_hyphen_bullet += 1

			if note == '':
				continue

			#remove 11.MRSA Colonization (report_id: 45075)
			words = note.split()
			lst = words[0].split('.')
			if len(lst) > 1:
				if lst[0].isdigit():
					note = lst[1] + ' '.join(words[1:])

			notes += [note]
			tags += [tag]
			notes_seg_idxs += [seg_idxs]
	
	if tags[-1] == ',':
		tags[-1] = '\n'

	#2) get statistic 
	n_lines = len(filter(lambda line: line.strip() != '', text.strip().split('\n')))
	n_notes = len(notes)
	n_period = 0
	n_bline = 0
	n_heading = 0
	n_bullet1 = 0
	max_word_len = 0
	max_byte_len = 0
	notes_bad = []
	prev_bad = 2
	for i in xrange(len(tags)):
		tag = tags[i]
		bad = 0
		if tag == '.':
			n_period += 1
			if prev_bad == 1:
				bad = 2

		elif tag == '\n':
			n_bline += 1
			note = notes[i]
			bytes = len(note)

			#Primary diagnoses: Fall, intracranial hemorrhage, facial
			if i > 0:
				(s, e) = dd['segment_spans'][notes_seg_idxs[i][-1]]
				if s > 0:
					if text[s - 1] == ' ':
						bytes += len(notes[i-1])

			if bytes > bad_len:
				bad = 1
			elif prev_bad == 1:
				bad = 2

			len1 = len(note.split())
			if max_word_len < len1:
				max_word_len = len1

			len2 = len(note)					
			if max_byte_len < len2:
				max_byte_len = len2

		elif tag == '_':
			n_heading += 1

		elif tag == '#':
			n_bullet1 += 1

		else:
			print "ERROR: unrecognized tag:", notes[i], [tag], ' idx:',idx
			sys.exit(0)
			
		notes_bad += [bad]
		prev_bad = bad

	n_bullet = n_bullet1
	n_comma = len(text.split(', ')) -1

	#3) return a dict dd
	dd['notes_tags'] = tags
	dd['notes'] = notes
	dd['notes_seg_idxs'] = notes_seg_idxs
	dd['notes_bad'] = notes_bad
	dd['info'] = {'n_lines':n_lines, 'n_notes':n_notes, 'n_period':n_period, 'n_bline':n_bline, \
		'n_comma':n_comma, 'n_heading':n_heading, 'n_bullet':n_bullet, \
		'n_hyphen_bullet':n_hyphen_bullet, 'max_word_len':max_word_len, 'max_byte_len':max_byte_len}

	return dd


#classify a dd to a type
def type_classifier(dd):	
	info = dd['info']

	[ n_lines, n_notes, n_period, n_bline, n_comma] = [info['n_lines'], info['n_notes'], \
			info['n_period'], info['n_bline'], info['n_comma'] ]
	[ n_heading, n_bullet, n_hyphen_bullet, max_word_len, max_byte_len ] = [info['n_heading'], \
			info['n_bullet'], info['n_hyphen_bullet'], info['max_word_len'], info['max_byte_len'] ]

	#use the correct # of bullet	
	if n_bullet == 0 and n_hyphen_bullet != 0:
		n_bullet = n_hyphen_bullet  

	#classify to a type
	if (n_notes - n_bullet - n_heading == n_period) \
	or (n_bullet == n_period and n_bline == 0)\
	or (n_bullet == n_bline and n_period == 0) :
		return 'good'
	
	elif (n_bullet != 0 ):
		
		if n_bullet == n_period + n_bline:
			return 'bullet_eq_period_bline' 

		elif n_period == n_bullet:
			return 'bullet_eq_period'
			
		else:
			return 'bullet_1'
			  
	elif n_period > 0:
		return 'period_1_bullet_0'
	
	#good too
	elif (n_notes - n_heading == n_heading or n_notes == 1) and n_comma == 0:
		return 'comma_0'
		
	#may be good
	elif (n_notes - n_heading == n_heading or n_notes == 1) and n_comma != 0:
		return 'comma_1'		

	#define a long/short sentence
	elif (n_period == 0 and n_bullet == 0 ):
		#may be good
		#if max_byte_len < 50:
		if sum(dd['notes_bad']) == 0:
			return 'period_0_bullet_0_short'			 
		#bad
		else:
			return 'period_0_bullet_0_long'
					
	else:
		print "ERROR: type_classifier()"
		sys.exit(0)

		
#------------------------------ print ---------------------------------------------------
#to string							 
def sentence2string(sentences, tags, with_index = True):
	j = 0
	tostring = ''

	for (sent, tag) in zip(sentences, tags):

		if tag == '_':
			tostring += '\n# ' + sent + '\n\n' 

		elif tag == '.':  # or tag == '\n':
			j += 1
			index = ''
			if with_index:
				index = '%d. '%(j)
			tostring += index + '%s.\n'%(sent)

			if tag == '\n':
				print "ERROR in sentence2string: unrecognized tag:", sent, [tag]; sys.exit(0)
			 

		elif tag == '#':
			continue

		else:
			print ("ERROR: unrecognized tag:", [tag])
			sys.exit(0)
		
	return tostring[:-1]
  
							 
#to a display string							 
def dd_tostring(notes, tags):
	j = 0
	tostring = ''
	added_index = False

	for (seg, tag) in zip(notes, tags):

		if tag == '_':
			tostring += seg + '\n' 

		elif tag == ',':
			if added_index:
				tostring += '%s, '%(seg)
			else:
				j += 1
				tostring += '%d. %s, '%(j, seg)
				added_index = True

		elif tag == ' ':
			if added_index:
				tostring += '%s '%(seg)
			else:
				j += 1
				tostring += '%d. %s '%(j, seg)
				added_index = True


		elif tag == '.' or tag == '\n':
			if added_index:
				tostring += '%s.\n'%(seg)
				added_index = False
			else:
				j += 1
				tostring += '%d. %s.\n'%(j, seg)

		elif tag == '#':
			continue
		
		else:
			print ("ERROR: unrecognized tag:", [tag])
			sys.exit(0)
		
	return '\n'.join(tostring.split('\n')[:-1])


#----------------------rule-based solver----------------------------------------------
#for dd with bullet points: assume that comma, linebreak and period are valid
def restore_bullet_2sentences(notes, notes_tags, notes_seg_idxs, segment_spans):
	sentences = []
	tags = []
	spans = []
	prev_tag = ''
	for i, (n, t, idxs) in enumerate(zip(notes, notes_tags, notes_seg_idxs)):

		if t != '#' :	
			s = segment_spans[idxs[0]][0]
			e = segment_spans[idxs[-1]][1]

		if t == '_':
			sentences += [n]
			tags += [t]
			spans += [(s, e)]
			prev_tag = t
			continue
			
		#print [prev_tag],[t],  n
		elif t == '#':
			prev_tag = t
			continue
			
		# t = ' ', '\n', '.'	
		if prev_tag == '#':
			sentences += [n]
			tags += ['.']
			spans += [(s, e)]
		else:
			if len(sentences) > 0 :
				sentences[-1] += ' ' + n
				tags[-1] = '.'   #t
				s = spans[-1][0]
				spans[-1] = (s, e)
				   
		prev_tag = t
		
	return sentences, tags, spans


#notes to sentences in a dd
def notes_2sentences(notes, notes_tags, notes_seg_idxs, segment_spans):
	sentences = []
	tags = []
	spans = []
	prev = ''
	prev_s = 10000000			
	for (note, t, idxs) in zip(notes, notes_tags, notes_seg_idxs):

		if t != '#' :	
			s = segment_spans[idxs[0]][0]
			e = segment_spans[idxs[-1]][1]

		if t == '_':
			sentences += [note]
			tags += [t]
			spans += [(s, e)]

		elif t == ',':
			if prev == '':
				prev = note + ', '
				prev_s = s
			else:
				prev = prev + note  + ', '
				if s < prev_s:
					prev_s = s		

		elif t == ' ':
			if prev == '':
				prev = note + ' '
				prev_s = s
			else:
				prev = prev + note + ' '
				if s < prev_s:
					prev_s = s
		  
		elif t == '\n' or t == '.': 
			if prev == '':
				sentences += [note]
				tags += ['.']  #[t]
				spans += [(s, e)]
			else:
				sentences += [prev + note]				
				tags += ['.'] 
				spans += [(prev_s, e)]
				prev = ''
				prev_s = 10000000
				continue
						
	return sentences, tags, spans
	

#for sentences with period: assume that comma and linebread are valid
def restore_period_2sentences(notes, notes_tags, notes_seg_idxs, segment_spans):
	sentences = []
	tags = []
	spans = []
	prev_tag = '#'
	for (note, t, idxs) in zip(notes, notes_tags, notes_seg_idxs):

		if t != '#' :	
			s = segment_spans[idxs[0]][0]
			e = segment_spans[idxs[-1]][1]

		if t == '_':
			sentences += [note]
			tags += [t]
			spans += [(s, e)]
			
		elif t == '\n' or t == '.': 
			if prev_tag == '\n':  # == ['\n', ' ']
				sentences[-1] += ' ' + note
				tags[-1] = '.'  #t
				prev_s = spans[-1][0]
				spans[-1] = (prev_s, e)

			else :
				sentences += [note]
				tags += ['.']  #[t]
				spans += [(s, e)]
			
		prev_tag = t
		
	return sentences, tags, spans


#incorrect comma to period
def comma2period(notes, notes_tags, notes_seg_idxs):
	ret_notes = []
	ret_tags = []
	ret_idxs = []
	for (note, t, idxs) in zip(notes, notes_tags, notes_seg_idxs):

		if t == '_':
			ret_notes += [note]
			ret_tags += [t]
			ret_idxs += [[idxs[0]]]
			
		elif t == '\n' or t == '.':
			segs = note.split(', ')
			ret_notes += [segs[0]]
			ret_tags += ['.']
			ret_idxs += [[idxs[0]]]
			if len(segs) > 1:
				for (seg, span) in zip(segs[1:], idxs[1:]):
					ret_notes += [seg]
					ret_tags += ['.']					
					ret_idxs += [[span]]
		
	return ret_notes, ret_tags, ret_idxs


#run testing
if __name__=="__main__":

	#1) get data-----------------
	import json
	with open ('dd_lst.json') as fj:
		dd_lst = json.load(fj)
		
	from dd_helper import *
		
	print "loading data:",len(dd_lst)
	import sys, string

	#good: notes_2sentences()
	good_dd = []
	comma_0_dd = []  #n_heading = n_sentence, with no comma

	#maybe good: restore_bullet_2sentences()
	bullet_eq_period_bline_dd = []  #n_period + n_bline = n_bullet
	bullet_eq_period_dd = [] #n_period = n_bullet and n_bline != 0
	period_1_bullet_0_dd =[]  #n_period > 0

	#bad: remove bullets and 
	bullet_1_dd = []  #mixed bullet and no bullet

	#good or bad: notes_2sentences()
	period_0_bullet_0_short_dd = []  #no period and no bullet with short length
	#good or bad: restore_comma_2sentences()
	comma_1_dd = []  #n_heading = n_sentence, with comma

	#bad: restore_linebreak_2sentences()
	period_0_bullet_0_long_dd = []  #no period and no bullet with long length
	bad_dd = []  #bad with no bullet

	learn_heading = []

	empty_dd = 0
	all_period = 0
	bullet_eq_period = 0
	bullet_eq_bline = 0
	sentence_1 = 0
	hyphen_bullet = 0
	has_period = 0

	period_lst_good = []
	comma_lst_good = []
	linebreak_lst_good = []
	period_lst_bad = []
	comma_lst_bad = []
	linebreak_lst_bad = []

	dd_notes_lst = []

	n_data = 0
	n_sample = 0
	n_non_sample = 0

	for i, d in enumerate(dd_lst): #[1165:1166]):   #[57:58]):   #[19574:19576]:
		idx = d['report_id']	

		page = d['discharge_diagnosis'].encode('utf-8')  #\
			#.replace('\n\n','\n').replace('\n\n','\n').replace('\n.','')\
			#.strip()
			
		#empty with '-'
		if page.translate(None, string.punctuation).strip() == '':
			empty_dd += 1
			continue

		dd = parse2segments(page)
		notes = segments2notes(dd, i, bad_len=48)		
		dd_type = type_classifier(notes)

		#print dd	
		dd.update({'dd_idx': i})
		dd.update({'report_id': idx})
		dd.update({'dd_type': dd_type})
		dd.update({'info': notes['info'], 'notes_bad': notes['notes_bad'], 'notes': notes['notes'], \
			'notes_tags': notes['notes_tags'], 'notes_seg_idxs': notes['notes_seg_idxs']})

		info = dd['info']		
		[ n_lines, n_notes, n_period, n_bline, n_comma] = [info['n_lines'], info['n_notes'], \
			info['n_period'], info['n_bline'], info['n_comma'] ]
		[ n_heading, n_bullet, n_hyphen_bullet, max_word_len, max_byte_len ] = [info['n_heading'], \
			info['n_bullet'], info['n_hyphen_bullet'], info['max_word_len'], info['max_byte_len'] ]

		#"""
		if n_bullet == 0 and n_hyphen_bullet != 0:
			n_bullet = n_hyphen_bullet
		if (n_notes - n_bullet - n_heading == n_period):
			all_period += 1	
		if (n_bullet == n_period and n_bline == 0):
			bullet_eq_period += 1
		if (n_bullet == n_bline and n_period == 0):
			bullet_eq_bline += 1
		if (n_notes - n_bullet - n_heading == n_heading or n_notes - n_bullet == 1): 
			sentence_1 += 1
		if n_period > 0:
			has_period += 1
		#""" 

		if dd_type == 'good':
			good_dd += [dd]

		#may be good  elif (n_bullet != 0 ):		
		elif dd_type == 'bullet_eq_period_bline':
			bullet_eq_period_bline_dd += [dd]

		elif dd_type == 'bullet_eq_period':
			bullet_eq_period_dd += [dd]

		elif dd_type == 'bullet_1':
			bullet_1_dd += [dd]

		elif dd_type == 'period_1_bullet_0':
			#may be good
			period_1_bullet_0_dd +=[dd]

		#good too
		elif dd_type == 'comma_0':
			comma_0_dd += [dd]

		#may be good
		elif dd_type == 'comma_1':
			comma_1_dd += [dd]

		#may be good
		elif dd_type == 'period_0_bullet_0_short':
			period_0_bullet_0_short_dd += [dd]

		#bad
		elif dd_type == 'period_0_bullet_0_long':
			period_0_bullet_0_long_dd += [dd]

		else:
			print "ERROR: dd_type!"
			sys.exit(0)

		dd_notes_lst += [dd]

	#1) dd_lst to dd_notes_lst  
	print "\n#1) dd_lst to dd_notes_lst ----------------------------------------" 
	print "done!"
	print len(dd_lst), len(dd_notes_lst)
	print "dd_notes_lst[0]:", json.dumps(dd_notes_lst[0], sort_keys=True,indent=2)


	#2) statistic: 
	print "\n#2) statiistic ----------------------------------------" 		
	"""
	good_dd = []
	comma_0_dd = []  #n_heading = n_sentence, with no comma

	#maybe good: restore_bullet_2sentences()
	bullet_eq_period_bline_dd = []  #n_period + n_bline = n_bullet
	bullet_eq_period_dd = [] #n_period = n_bullet and n_bline != 0
	period_1_bullet_0_dd =[]  #n_period > 0

	#bad: remove bullets and 
	bullet_1_dd = []  #mixed bullet and no bullet

	#good or bad: notes_2sentences()
	period_0_bullet_0_short_dd = []  #no period and no bullet with short length
	#good or bad: restore_comma_2sentences()
	comma_1_dd = []  #n_heading = n_sentence, with comma

	#bad: restore_linebreak_2sentences()
	period_0_bullet_0_long_dd = []  #no period and no bullet with long length
	bad_dd = []  #bad with no bullet
	"""
	print "\nstatistic:"
	print "samples		 :",len(dd_lst)
	print "good_dd		 : %d (%d, %d)"%(len(good_dd)+len(comma_0_dd), len(good_dd), \
		len(comma_0_dd))
	print "possible_good_dd: %d (%d, %d, %d, %d)"%(len(bullet_eq_period_bline_dd)+\
		len(bullet_eq_period_dd)+len(period_1_bullet_0_dd)+len(bullet_1_dd), \
		len(period_1_bullet_0_dd),len(bullet_eq_period_bline_dd),len(bullet_eq_period_dd),\
		len(bullet_1_dd))
	print "good_bad_dd	 : %d (%d, %d)"%(len(period_0_bullet_0_short_dd)+len(comma_1_dd),\
		len(period_0_bullet_0_short_dd), len(comma_1_dd) )
	print "bad_dd		  : %d (%d, %d)"%(len(period_0_bullet_0_long_dd)+len(bad_dd),\
		len(period_0_bullet_0_long_dd),len(bad_dd) )

	print empty_dd, all_period, bullet_eq_period, bullet_eq_bline, sentence_1, has_period


	#3) restore_bullet_2sentences(notes, notes_tags)  
	print "\n#3) restore_bullet_2sentences() ----------------------------------------" 
	for dd in bullet_1_dd[0:3]:  

		#dd = dd_notes_lst[d['dd_idx']]		
		sentences, tags, spans = restore_bullet_2sentences(dd['notes'], dd['notes_tags'], \
			dd['notes_seg_idxs'], dd['segment_spans'])
		
		print '\n================\nreport_id:%s\n================(%d, %s)'%(dd['report_id'],\
			dd['dd_idx'], dd_type)
		print '------------input-------------\n', dd_lst[dd['dd_idx']]['discharge_diagnosis']
		print '------------output------------\n', sentence2string(sentences, tags)

	#4) restore_period_2sentences(notes, notes_tags, notes_bad, n_period, n_bline)
	print "\n#4) restore_period_2sentences()--------------------------------------"  
	for dd in period_1_bullet_0_dd[1000:1003]:

		sentences, tags, spans = restore_period_2sentences(dd['notes'], dd['notes_tags'], \
			dd['notes_seg_idxs'], dd['segment_spans'])
		
		print '\n================\nreport_id:%s\n================(%d, %s)'%(dd['report_id'],\
			dd['dd_idx'], dd_type)
		print '------------input-------------\n', dd_lst[dd['dd_idx']]['discharge_diagnosis']
		print '------------output------------\n', sentence2string(sentences, tags)

	#5) restore_comma_2sentences(notes, notes_tags)
	print "\n#5) restore_comma_2sentences(notes, notes_tags)----------------------------------------"
	for dd in comma_1_dd[1000:1003]:

		notes, tags, idxs = comma2period(dd['notes'], dd['notes_tags'], dd['notes_seg_idxs'])
		sentences, tags, spans = notes_2sentences(notes, tags, idxs, dd['segment_spans'])	

		print '\n================\nreport_id:%s\n================(%d, %s)'%(dd['report_id'],\
			dd['dd_idx'], dd_type)
		print '------------input-------------\n', dd_lst[dd['dd_idx']]['discharge_diagnosis']
		print '------------output------------\n', sentence2string(sentences, tags)


	data = """
Example 1:
1. Atrial fibrillation.
2. Hypertension.
Appointment on Monday 22/7/17:
3. Diabetes

Example 2:
Atrial fibrillation  , hypertension, diabetes

Example 3:
-Atrial fibrillation
-hypertension
-Diabetes

Example 4:
Primary
1. Atrial	  Fibrillation
2. Hypertension
Secondary
1. Diabetes

Example 5:
Primary
1. Atrial Fibrillation, Hypertension
1. Diabetes. stage 2-3
Secondary
3). Diabetes
13. Hypertension
""" 
	#6) example
	print "\n#6) example ----------------------------------------"
	#rule-based   
	for i, page in enumerate(data.strip().split('\n\n')[:5]):
		idx = i
		base = 0
		original_idx = i

		dd = parse2segments(page)		
		notes = segments2notes(dd, i, bad_len=48)		
		dd_type = type_classifier(notes)

		#print dd	
		dd.update({'dd_idx': i})
		dd.update({'report_id': idx})
		dd.update({'dd_type': dd_type})
		dd.update({'info': notes['info'], 'notes_bad': notes['notes_bad'], 'notes': notes['notes'],\
				'notes_tags': notes['notes_tags'], 'notes_seg_idxs': notes['notes_seg_idxs']})

		if dd_type == 'good' \
		or dd_type == 'bullet_eq_period_bline' \
		or dd_type == 'comma_0':
			sentences, tags, spans = notes_2sentences(dd['notes'], dd['notes_tags'], \
				dd['notes_seg_idxs'], dd['segment_spans'])

		elif dd_type == 'bullet_eq_period' \
		or dd_type == 'bullet_1':
			sentences, tags, spans = restore_bullet_2sentences(dd['notes'], dd['notes_tags'], \
				dd['notes_seg_idxs'], dd['segment_spans'])

		elif dd_type == 'period_1_bullet_0':
			sentences, tags, spans = restore_period_2sentences(dd['notes'], dd['notes_tags'], \
				dd['notes_seg_idxs'], dd['segment_spans'])

		#may be good
		elif dd_type == 'comma_1':
			new_notes, tags, idxs = comma2period(dd['notes'], dd['notes_tags'], dd['notes_seg_idxs'])
			sentences, tags, spans = notes_2sentences(new_notes, tags, idxs, dd['segment_spans'])

		#may be good
		#if max_word_len < 8 and max_byte_len < 50:
		#if max_byte_len < 50:
		elif dd_type == 'period_0_bullet_0_short':
			sentences, tags, spans = notes_2sentences(dd['notes'], dd['notes_tags'], \
				dd['notes_seg_idxs'], dd['segment_spans']) 

		#bad
		elif dd_type == 'period_0_bullet_0_long':	  
			sentences, tags, spans = notes_2sentences(dd['notes'], dd['notes_tags'], \
				dd['notes_seg_idxs'], dd['segment_spans'])	 
		else:
			print "ERROR: dd_type!"
			sys.exit(0)

		dd_notes_lst += [dd]
		original_spans = [(s + base, e + base) for (s, e) in spans]

		#print dependent on dd_type
		if 1 == 1:
	#	if dd_type == 'good':
	#	if dd_type == 'bullet_eq_period_bline':
	#	if dd_type == 'bullet_eq_period':
	#	if dd_type == 'bullet_1':
	#	if dd_type == 'period_1_bullet_0':
	#	if dd_type == 'comma_0':
	#	if dd_type == 'comma_1': 
	#	if dd_type == 'period_0_bullet_0_short' and dd['info']['n_comma'] > 0:
	#	if dd_type == 'period_0_bullet_0_long':

			print '\n================\nreport_id:%s\n================(%d, %s)'%(dd['report_id'],\
				dd['dd_idx'], dd_type)
			print '------------input-------------\n', page
			print '------------output------------\n', sentence2string(sentences, tags)
			print spans

#from __future__ import print_function
import time
import json
from dd_helper import *
from dd_model_helper import *

input_default='/home/lca80/Desktop/data/emtell/sept2017/all_m3_CT_and_DS_reports.csv'
num_test = 3000000   #1000   #

def restore_punct(dd_lst, model_dir):
	comma_clf = load_model(model_dir + 'comma_clf.model')
	linebreak_clf = load_model(model_dir + 'linebreak_clf.model')

	t0 = time.time()
	dd_sentences_lst = []

	for i, d in enumerate(dd_lst[:num_test]): 
		idx = d['report_id']        
		page = d['discharge_diagnosis'].encode('utf-8')
		base = d['original_file_span_base'] 
		original_idx = d['original_file_idx']
		file_name = d['original_file']

		dd = parse2segments(page)   
		notes = segments2notes(dd, i, bad_len=48)    
		dd_type = type_classifier(notes)

		#print dd    
		dd.update({'dd_idx': i})
		dd.update({'report_id': idx})
		dd.update({'dd_type': dd_type})
		dd.update({'info': notes['info'], 'notes_bad': notes['notes_bad'], 'notes': notes['notes'], \
			'notes_tags': notes['notes_tags'], 'notes_seg_idxs': notes['notes_seg_idxs']})
	    

		if dd_type == 'good' \
		or dd_type == 'bullet_eq_period_bline' \
		or dd_type == 'comma_0':
			sentences, tags, spans = notes_2sentences(dd['notes'], dd['notes_tags'], \
				dd['notes_seg_idxs'], dd['segment_spans'])

		elif dd_type == 'bullet_eq_period' \
		or dd_type == 'bullet_1':
			sentences, tags, spans = restore_bullet_2sentences(dd['notes'], dd['notes_tags'],\
				dd['notes_seg_idxs'], dd['segment_spans'])
		      
		elif dd_type == 'period_1_bullet_0':
			sentences, tags, spans = restore_period_2sentences(dd['notes'], dd['notes_tags'], \
				dd['notes_seg_idxs'], dd['segment_spans'])
	       
		#may be good
		elif dd_type == 'comma_1':
			#new_notes, tags, idxs = restore_comma_2sentences(dd['notes'], dd['notes_tags'], \
			#    dd['notes_seg_idxs'], comma_clf)  
			new_notes, tags, idxs = comma2period(dd['notes'], dd['notes_tags'], dd['notes_seg_idxs'])
			sentences, tags, spans = notes_2sentences(new_notes, tags, idxs, dd['segment_spans'])
		
		#may be good
		#if max_byte_len < 50:
		elif dd_type == 'period_0_bullet_0_short':
			"""     
			new_nodes, tags, idxs = restore_linebreak_by_clf(dd['notes'], dd['notes_tags'], \
				dd['notes_bad'], dd['notes_seg_idxs'], linebreak_clf)        
			new_nodes, tags, idxs = restore_comma_by_clf(new_nodes, tags, idxs, comma_clf)
			sentences, tags, spans = notes_2sentences(new_notes, tags, idxs, dd['segment_spans'])
			#"""
			sentences, tags, spans = notes_2sentences(dd['notes'], dd['notes_tags'], \
				dd['notes_seg_idxs'], dd['segment_spans'])
			#""" 
		
		#bad
		elif dd_type == 'period_0_bullet_0_long':      
			new_notes, tags, idxs = restore_linebreak_by_clf(dd['notes'], dd['notes_tags'], \
				dd['notes_bad'], dd['notes_seg_idxs'], linebreak_clf)           
			new_notes, tags, idxs = restore_comma_by_clf(new_notes, tags, idxs, comma_clf)       
			sentences, tags, spans = notes_2sentences(new_notes, tags, idxs, dd['segment_spans'])      
		else:
			print "ERROR: dd_type!"
			sys.exit(0)
	    
		#dd_notes_lst += [dd]

		original_spans = [(s + base, e + base) for (s, e) in spans]
		dd_sentences_lst += [{'report_id': idx, 'sentences': sentences, 'tags': tags,'spans': spans, \
			'discharge_diagnosis': page, 'original_spans': original_spans, \
			'original_file': file_name, 'original_file_idx': original_idx}]
	 
		#print if dd_type
		if 1 == 0:
		#if dd_type == 'good':
		#if dd_type == 'bullet_eq_period_bline':
		#if dd_type == 'bullet_eq_period':
		#if dd_type == 'bullet_1':
		#if dd_type == 'period_1_bullet_0':
		#if dd_type == 'comma_0':
		#if dd_type == 'comma_1': 
		#if dd_type == 'period_0_bullet_0_short' and dd['info']['n_comma'] > 0:
		#if dd_type == 'period_0_bullet_0_long':

			print '\n================\nreport_id:%s\n================(%d, %s)'%\
				(dd['report_id'],dd['dd_idx'], dd_type)
			print '------------input-------------\n', d['discharge_diagnosis']
			print '------------output------------\n', sentence2string(sentences, tags)
			print original_spans

	print "time:", time.time() - t0
	print len(dd_lst), len(dd_sentences_lst)

	return dd_sentences_lst


if __name__ == '__main__':
	import argparse
	import json
	import os

	argparser = argparse.ArgumentParser()

	argparser.add_argument("-i", "--input_csv", dest="input_csv", default=input_default, \
		help="input report data csv file path (default={})".format(input_default))
	argparser.add_argument("-d", "--input_dd_path", dest="input_dd_path", default="dd_lst.json", \
		help="input discharge diagnoses data json file path (default=dd_report_lst.json)")
	argparser.add_argument("-o", "--output_path", dest="output_path", default="dd_sentences_lst.json", \
		help="output json file path (default=dd_sentences_lst.json)")
	argparser.add_argument("-m", "--model_dir", dest="model_dir", default="model/", \
		help="models directory                (default=model/)")
	args = argparser.parse_args()

	try:
		with open ('dd_lst.json') as fj:
		    dd_lst = json.load(fj)
	except:
		data_path = args.input_csv
		file_name = data_path.split('/')[-1]
		dd_lst, _ = extract_dd_raw(data_path)

	#print json.dumps(dd_lst[520], sort_keys=True,indent=2)
	dd_sentences_lst = restore_punct(dd_lst, args.model_dir)
	with open('dd_sentences_lst.json', 'w') as fj:
		json.dump(dd_sentences_lst, fj)
	print "\ndischarge diagnoses are saved to dd_sentences_lst.json"
	print "Done!"
	

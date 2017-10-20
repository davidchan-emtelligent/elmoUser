#from __future__ import print_function
import time
import argparse
import json
import os
from dd_helper import *

input_default='/home/lca80/Desktop/data/emtell/sept2017/all_m3_CT_and_DS_reports.csv'

if __name__ == '__main__':

	argparser = argparse.ArgumentParser()

	argparser.add_argument("-i", "--input_csv", dest="input_csv", default=input_default, \
		help="input csv report data file path (default={})".format(input_default))
	argparser.add_argument("-d", "--input_dd_path", dest="input_dd_path", default="dd_lst.json", \
		help="input discharge diagnosis path  (default=dd_report_lst.json)")
	argparser.add_argument("-o", "--output_path", dest="output_path", default="dd_sentences_lst.json", \
		help="output path                     (default=dd_sentences_lst.json)")
	argparser.add_argument("-m", "--model_dir", dest="model_dir", default="model/", \
		help="models directory                (default=model/)")
	args = argparser.parse_args()

	reports_lst = []
	try:
		with open ('dd_lst.json') as fj:
		    dd_lst = json.load(fj)
	except:
		data_path = args.input_csv
		file_name = data_path.split('/')[-1]
		dd_lst, reports_lst = extract_dd_raw(data_path)

	dd_sentences_lst = get_restored_sentences(dd_lst)

	#"""
	for (i, dd) in map(lambda i: (i, dd_sentences_lst[i]), [520,752,766,820]):
		print '\n================\nreport_id:%s\n================(%s index:%d)'%\
			(dd['report_id'], dd['original_file'], dd['original_file_idx'])
		print '------------\x1b[0;37;44m input \x1b[0m-------------\n', dd['discharge_diagnosis']
		print '------------\x1b[0;37;41m output \x1b[0m------------\n', sentence2string(dd['sentences'], \
			dd['sentences_tags'])
		print dd['original_spans']
	    
	(s , e) = dd['original_spans'][-1]
	if reports_lst != []:    
		print [s, e], reports_lst[dd['original_file_idx']]['original_report'][s:e]
	#"""

	with open(args.output_path, 'w') as fj:
		json.dump(dd_sentences_lst, fj)
	print "\ndischarge diagnoses are saved to %s"%(args.output_path)
	print "Done!"
	

from __future__ import print_function
import unicodecsv as csv
import sys, bz2
import re

data_path = '/Desktop/data/emtell/sept2017/all_m3_CT_and_DS_reports.csv'

# load up the nltk tokenizer
#from nltk.tokenize import regexp_tokenize

# before you can run the following code to use the StanfordTokenizer
# you need to `brew install stanford-parser` on macosx or install
# from source on linux. also you should have the Anaconda distribution
# of Python for nltk
#from nltk.tokenize.stanford import StanfordTokenizer
#tokenizer = StanfordTokenizer(path_to_jar="/usr/local/Cellar/stanford-parser/3.5.2/libexec/stanford-parser.jar")

class CsvWriter:
    def __init__(self, file, headers, append=False):
        self.file = file
        self.outputf = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
        if not append:
            self.outputf.writerow(headers)

    def write(self, row):
        self.outputf.writerow(row)

    def close(self):
        self.file.close()

class CsvReader:

    def __init__(self, opts):
        # filenames
        self.inputfile = opts.inputfile
        try:
            self.outputfile = opts.outputfile
        except AttributeError:
            self.outputfile = None

        # streaming writer
        self.outputf = None
        self.writer = None

        # uses header from the CSV file if True
        self.has_header = opts.has_header 

        # data storage
        self.header = []

        # set up csv
        csv.field_size_limit(sys.maxsize)

        # set up matching
        try:
            self.match_field = str(opts.match_field)
        except AttributeError:
            self.match_field = None

        try:
            self.match_re = re.compile(opts.match_re_pat, re.MULTILINE | re.DOTALL) if opts.match_re_pat is not None else None
        except AttributeError:
            self.match_re = None

        # number of lines to print
        try:
            self.num = int(opts.num)
        except AttributeError:
            self.num = 0

        # print progress after these many lines
        try:
            self.p = int(opts.p)
        except AttributeError:
            self.p = 10000

        # for nltk tokenizer if useful later
        #self.tokenize_pattern = '\w+|\$[\d\.]+|\S+'

    def __del__(self):
        if self.outputf is not None:
            self.outputf.close()

    def read_CSV(self, callback):
        data = None
        create_header = True
        retval = True
        csvfile = None
        if (len(self.inputfile) > 5) and (self.inputfile[-4:] == ".bz2"):
            csvfile = bz2.BZ2File(self.inputfile, 'rU')
            sys.stderr.write("opening bzip2 file {}\n".format(self.inputfile))
        else:
            csvfile = open(self.inputfile, 'rU')
            sys.stderr.write("opening csv file {}\n".format(self.inputfile))
        line_number = 0
        output_line_number = 0
        callback_successful = 0
        if csvfile:
            try:
                inputf = csv.reader(csvfile, dialect='excel', quotechar='"', escapechar='\\')
                for row in inputf:
                    line_number += 1
                    if line_number % self.p == 0:
                        sys.stderr.write(".")
                    if create_header:
                        # check if the user has indicated that a header exists as the first row
                        # if so, use it as the header array else create a header array with index numbers
                        create_header = False
                        if self.has_header:
                            line_number -= 1
                            self.header = row
                            continue
                        else:
                            self.header = [str(i) for i in range(len(row))]
                            data = { self.header[idx]: value for (idx, value) in enumerate(row) }
                    elif self.match_re is not None and not self.match_re.search(row[self.header.index(self.match_field)]):
                        continue
                    else:
                        output_line_number += 1
                        if (self.num > 0) and (output_line_number > self.num):
                            line_number -= 1
                            break
                        data = { self.header[idx]: value for (idx, value) in enumerate(row) }
                    try:
                        if callback(data):
                            callback_successful += 1
                    except Exception as e:
                        sys.stderr.write("Error: {0}\n".format(e.message))
                        sys.stderr.write("Callback function failed on line: {}\n".format(line_number))
                        sys.stderr.write("{}\n".format(data))
                        retval = False
                sys.stderr.write("\n")
            except Exception as e:
                sys.stderr.write("Error: {0}\n".format(e.message))
                sys.stderr.write("File open for file: {}\n".format(self.inputfile))
                sys.exit(1)
            finally:
                csvfile.close()
        sys.stderr.write("Finished reading {} lines from file: {}\n".format(line_number, self.inputfile))
        sys.stderr.write("Callback function successful for {} lines\n".format(callback_successful))
        return retval

    def writer_callback(self, data, new_header):
        if self.outputf is None:
            try:
                self.outputf = open(self.outputfile, 'wb')
            except:
                sys.stderr.write("could not open file {}\n".format(self.outputfile))
                sys.exit(1)
            if len(self.header) == 0:
                sys.stderr.write("Error: header is empty\n")
                sys.exit(1)
            if new_header is not None:
                merged_header = self.header + [new_header]
            else:
                merged_header = self.header
            self.writer = csv.DictWriter(self.outputf, fieldnames=merged_header, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
            self.writer.writeheader()
        if self.writer is not None:
            self.writer.writerow(data)
        else:
            raise ValueError("Writer should not be empty")
        return True

    def new_header_callback(self, data, new_header_list):
        if self.outputf is None:
            try:
                self.outputf = open(self.outputfile, 'wb')
            except:
                sys.stderr.write("could not open file {}\n".format(self.outputfile))
                sys.exit(1)
            if new_header_list is None or len(new_header_list) == 0:
                sys.stderr.write("Error: header is empty\n")
                sys.exit(1)
            self.writer = csv.DictWriter(self.outputf, fieldnames=new_header_list, quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')
            self.writer.writeheader()
        if self.writer is not None:
            self.writer.writerow(data)
        else:
            raise ValueError("Writer should not be empty")
        return True

    def dummy_callback(self, data):
        return self.writer_callback(data, None)

    def make_writer(self, output_header, output_file_name, append=False):
        outputfile = open(output_file_name, ('a' if append else 'w') +'b')
        return CsvWriter(outputfile, output_header, append=append)

    def write_CSV(self, output_header, output_rows, output_file_name):
        with open(output_file_name, 'wb') as outputfile:
            writer = CsvWriter(outputfile, output_header)
            for output_row in output_rows:
                writer.write(output_row)
        return True

if __name__ == '__main__':
    import argparse
    import json
    import os

    argparser = argparse.ArgumentParser()
    if 'HOME' in os.environ:
        default_inputfile = os.environ['HOME'] + data_path
    else:
        default_inputfile = '../output/all_rad_sorted.csv.bz2'
    argparser.add_argument("-i", "--inputfile", dest="inputfile", default=default_inputfile,
        help="input filename (default={})".format(default_inputfile))
    argparser.add_argument("-o", "--outputfile", dest="outputfile", default='../output/all_rad_output.csv',
        help="input filename (default=../output/all_rad_output.csv)")
    argparser.add_argument("-n", "--num", dest="num", type=int, default=0,
        help="number of lines to print (default=0 which means print all)")
    argparser.add_argument("-p", "--progress", dest="p", type=int, default=10000,
        help="print progress after these many lines (default=10000)")
    argparser.add_argument("-r", "--header", dest="has_header", action="store_false", default=True,
        help="csv file does not have a header line with column names")
    argparser.add_argument("-f", "--field", dest="match_field", type=str, default='sentence_text',
        help="field for searches default: sentence_text (name if using --header, otherwise index)")
    argparser.add_argument("-m", "--match", dest="match_re_pat", type=str, default=None,
        help="show only records where the specified field matches a given regular expression")
    args = argparser.parse_args()

    c = CsvReader(args)
    #c.read_CSV(lambda x: print(json.dumps(x)))
    c.read_CSV(c.dummy_callback)
    #amount = "all" if args.num == 0 else args.num
    #sys.stderr.write("wrote {} lines from file: {} to file: {}\n".format(amount, args.inputfile, args.outputfile))

import os

path_data = "/shared/dropbox/ctakes_conll/tokenized_text"

ds = os.listdir(path_data)

n_tokens = 0
for d in ds:
	fs = os.listdir(os.path.join(path_data, d))
	text = ""
	for f in fs:
		with open(os.path.join(path_data, d, f), "r") as fd:
			text += fd.read().strip() + '\n'
	tokens = text.split()
	len1 = len(tokens)
	print (d, len1)
	n_tokens += len1

print (n_tokens)

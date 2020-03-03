from transformers import BertTokenizer
import argparse
import os


tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')





parser = argparse.ArgumentParser()
parser.add_argument('--delimiter', type=str, default='\t')
parser.add_argument('--folder', type=str)
parser.add_argument('--fname', type=str)
parser.add_argument('--max_len', type=int)
args = parser.parse_args()

in_fname = args.folder + args.fname
base = os.path.splitext(in_fname)[0]
out_fname = base + '.tokenized.tsv'

with open(out_fname, 'w') as out_f:
    with open(in_fname, 'r') as in_f:
        #ids = list()
        count = 0
        data = {}

        
        line = in_f.readline()
        while line:

            if count % 10000 == 0:
                print(f'lines read: {count}')
            if line != '':
                id_, text = line.strip().split(args.delimiter, 1)
                tokenized_ids = tokenizer_bert.encode(text, max_length = args.max_len)
                out_f.write(id_ + ' ' + ' '.join(str(t) for t in tokenized_ids) + '\n')
                count += 1
            line = in_f.readline()

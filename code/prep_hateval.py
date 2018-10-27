from to_tokens import Tokenizer
import os
import csv

DEBUG = True
base_path = '../data/hateval/public_development_es/'
proc_path = '../data/processed/'
languages = ['es','en']
tokenizer_options = {
    'delete_punctuation':True, 
    'delete_urls':True, 
    'delete_mentions':True, 
    'delete_hashtags':True,
} 


for filename in os.listdir(base_path):
    if not filename.endswith('.tsv'):
        continue
    # read all the content of the text column and the class
    language = filename[:2]
    if language not in languages:
        continue
    suffix = filename[2:-4]
    tokenizer = Tokenizer(languages=[language])
    X, Y = [], []
    with open(os.path.join(base_path, filename)) as tsv_file:
        first = True
        for line in tsv_file:
            if first:
                first = False
                continue
            data = line[:-1].split('\t')
            text, label = data[1], data[2]
            X.append(text)
            Y.append(label)
    X_tokens = tokenizer.tokenize_list_of_texts(X, language, **tokenizer_options)
    out_X_file = language + '_X' + suffix + '.txt'
    out_Y_file = language + '_Y' + suffix + '.txt'
    for out_file, data in [(out_X_file, X_tokens), (out_Y_file, Y)]:
        with open(os.path.join(proc_path, out_file), 'w') as txt_out_file:
            for dat in data:
                to_write = '\t'.join(dat)
                txt_out_file.write(to_write + '\n')



from to_tokens import Tokenizer
import os
import csv

DEBUG = True
base_path = '../data/raw/'
proc_path = '../data/processed/'
languages = ['es','en','it']
tokenizer_options = {
    'delete_punctuation':True, 
    'delete_urls':True, 
    'delete_mentions':True, 
    'delete_hashtags':True,
} 


for filename in os.listdir(base_path):
    if not filename.endswith('.csv'):
        continue
    # read all the content of the text column and the class
    language = filename[:2]
    if language not in languages:
        continue
    suffix = filename[2:-4]
    tokenizer = Tokenizer(languages=[language])
    X, Y = [], []
    with open(os.path.join(base_path, filename)) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            text, label = row['text'], row['class']
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






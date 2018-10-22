vec_base_path = '/data/word_embeddings/multilingual/fasttext/small/'
#vec_base_path = '/Users/jperez/research/nlp/word-embeddings/multilingual/'
data_base_path = '../data/processed/'
format_name = 'wiki.{language}.{size}.align.vec'
format_dataset = '{label}_{XorY}_{settype}.txt'
history_path = 'train_history/'
model_path = 'best_models/'
results_file = 'results_{timeday}.txt'

vector_size = 300

languages = ['es', 'en', 'it']
size_vec_file = '500k'

labels_train_data = ['es', 'en', 'it']
labels_test_data = ['es', 'en', 'it', 'es_manual']

labels_to_language = {
    'es': 'es',
    'en': 'en',
    'it': 'it',
    'es_manual': 'es',
}

from collections import defaultdict
from datetime import datetime

config_data = defaultdict(dict)
vector_files = defaultdict(str)

timeday = str(datetime.now().date()) + '_' + str(datetime.now().time())
results_file = results_file.format(timeday=timeday)

for language in languages:
    vector_files[language] = vec_base_path + format_name.format(language=language, size=size_vec_file)

for label in (set(labels_train_data) | set(labels_test_data)):
    config_data[label] = defaultdict(dict)
    for settype in ['train', 'test']:
        config_data[label][settype] = defaultdict(dict) 

for label in labels_to_language:
    config_data[label]['language'] = labels_to_language[label]

for label in labels_train_data:
    config_data[label]['train']['X'] = data_base_path + format_dataset.format(label=label, XorY='X', settype='train')
    config_data[label]['train']['Y'] = data_base_path + format_dataset.format(label=label, XorY='Y', settype='train')

for label in labels_test_data:
    config_data[label]['test']['X'] = data_base_path + format_dataset.format(label=label, XorY='X', settype='test')
    config_data[label]['test']['Y'] = data_base_path + format_dataset.format(label=label, XorY='Y', settype='test')

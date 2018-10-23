vec_base_path = '/data/word_embeddings/multilingual/fasttext/'
#vec_base_path = '/Users/jperez/research/nlp/word-embeddings/multilingual/'
data_base_path = '../data/processed/'
format_name = 'wiki.{language}.align.vec'
format_dataset = '{label}_{XorY}_{settype}.txt'
history_path = 'train_history/'
model_path = 'best_models/models_{timeday}/'
results_file = 'results/results_{timeday}.txt'
config_file = 'results/results_{timeday}.config.txt'

vector_size = 300
limit_vectors = 1000
batch_size = 32

num_threads = 1

languages = ['es', 'en', 'it']

labels_train_data = ['es', 'en', 'it']
labels_test_data = ['es', 'en', 'it', 'es_manual']

labels_to_language = {
    'es': 'es',
    'en': 'en',
    'it': 'it',
    'es_manual': 'es',
}


#############################
#############################
#############################



from collections import defaultdict
from datetime import datetime
import os

config_data = defaultdict(dict)
vector_files = defaultdict(str)

timeday = datetime.now().strftime('%Y_%m_%d_%H_%M')
results_file = results_file.format(timeday=timeday)
config_file = config_file.format(timeday=timeday)
model_path = model_path.format(timeday=timeday)
if not os.path.exists(model_path):
    os.makedirs(model_path)

for language in languages:
    vector_files[language] = vec_base_path + format_name.format(language=language)

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

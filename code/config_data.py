#vec_base_path = '/data/word_embeddings/multilingual/fasttext/'
vec_base_path = '/Users/jperez/research/nlp/word-embeddings/multilingual/'
data_base_path = '../data/processed/'
format_name = 'wiki.{language}.align.vec'
format_dataset = '{label}_{XorY}_{settype}.txt'
history_path = 'train_history/history_{timeday}/'
model_path = 'best_models/models_{timeday}/'
results_file = 'results/results_{timeday}.txt'
results_config_file = 'results/results_{timeday}.config.txt'

vector_size = 300
num_threads = 1

languages = ['es', 'en', 'it']

labels_train_data = ['es', 'en', 'it', 'es_se', 'en_se']
labels_dev_data = ['es_se', 'en_se']
labels_test_data = [
    'es', 
    'en', 
    'it', 
    'es_manual',
    'es-en',
    'es-en_manual',
    'it-en',
    'en-es',
    'it-es',
    'en-it',
    'es-it',
]

labels_to_language = {
    'es': 'es',
    'en': 'en',
    'it': 'it',
    'es_manual': 'es',
    'es_se': 'es',
    'en_se': 'en',
    'es-en': 'en',
    'es-en_manual': 'en',
    'it-en': 'en',
    'en-es': 'es',
    'it-es': 'es',
    'en-it': 'it',
    'es-it': 'it',
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
results_config_file = results_config_file.format(timeday=timeday)
model_path = model_path.format(timeday=timeday)
if not os.path.exists(model_path):
    os.makedirs(model_path)
history_path = history_path.format(timeday=timeday)
if not os.path.exists(history_path):
    os.makedirs(history_path)

for language in languages:
    vector_files[language] = vec_base_path + format_name.format(language=language)

for label in (set(labels_train_data) | set(labels_test_data)):
    config_data[label] = defaultdict(dict)
    for settype in ['train', 'test']:
        config_data[label][settype] = defaultdict(dict) 

for label in labels_to_language:
    config_data[label]['language'] = labels_to_language[label]

for labels_data, settype in [(labels_train_data, 'train'),(labels_dev_data, 'dev'),(labels_test_data, 'test')]:
    for label in labels_data:
        config_data[label][settype]['X'] = data_base_path + format_dataset.format(label=label, XorY='X', settype=settype)
        config_data[label][settype]['Y'] = data_base_path + format_dataset.format(label=label, XorY='Y', settype=settype)     

# for label in labels_train_data:
#     config_data[label]['train']['X'] = data_base_path + format_dataset.format(label=label, XorY='X', settype='train')
#     config_data[label]['train']['Y'] = data_base_path + format_dataset.format(label=label, XorY='Y', settype='train')

# for label in labels_test_data:
#     config_data[label]['test']['X'] = data_base_path + format_dataset.format(label=label, XorY='X', settype='test')
#     config_data[label]['test']['Y'] = data_base_path + format_dataset.format(label=label, XorY='Y', settype='test')

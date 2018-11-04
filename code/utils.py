import torch
import numpy as np
import hashlib
from torch.utils.data import Dataset, ConcatDataset, random_split, DataLoader
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from config_data import config_data, vector_files
import pickle
import sys
from datetime import datetime
from config_data import vector_size
import ipdb

_hash_len = 8

sklearn_options = {
    'accuracy': accuracy_score,
    'f1': f1_score,
    'recall': recall_score,
    'precision': precision_score
}

def get_hyperparameter_options(hp_file):
    d = open(hp_file).read()
    d = d.split('$')
    options = [eval(s) for s in d]
    return options

def reset_all_seeds(RANDOM_SEED=33):
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

def score(Y_pred_prob, Y, sklearn_metric=accuracy_score):
    if type(sklearn_metric) == str:
        sklearn_metric = sklearn_options[sklearn_metric]
    Y_pred = (Y_pred_prob.float() >= 0.5).float()
    Y_pred = Y_pred.cpu().view(-1).numpy()
    Y = Y.cpu().view(-1).numpy()
    return sklearn_metric(Y, Y_pred)

def report(Y_pred_prob, Y):
    Y_pred = (Y_pred_prob >= 0.5).float()
    Y_pred = Y_pred.cpu().view(-1).numpy()
    Y = Y.cpu().view(-1).numpy()
    target_names = ['class_0', 'class_1']
    return classification_report(Y, Y_pred, target_names=target_names) 

def hash_texto(texto):
    out = hashlib.sha224(texto.encode('utf-8')).hexdigest()
    return str(out)[:_hash_len]

def load_scenarios(scenarios_file):
    with open(scenarios_file) as file:
        scenarios = []
        for line in file:
            scenario = eval(line[:-1])
            scenarios.append(scenario)
    return scenarios

def save_history(history_path, scenario, model):
    outname = hash_texto(str(scenario)) + '.hist'
    with open(history_path + outname, 'w') as outfile:
        outfile.write(str(scenario) + '\n')
        outfile.write(str(model.best_dev_acc['value']) + ' ') 
        outfile.write(str(model.best_dev_loss['value'])  + '\n')
        outfile.write(str(model.history_output) + '\n')

def save_config(model, hps, config_file):
    with open(config_file, 'w') as outfile:
        outfile.write(str(hps) + '\n\n')
        outfile.write('vector_size:' + str(vector_size) + '\n')
        outfile.write('batch_size:' + str(hps['batch_size']) + '\n')
        outfile.write('limit_vectors:' + str(hps['limit_vectors']) + '\n')
        outfile.write('\n')
        outfile.write(str(model.net) + '\n\n')
        outfile.write(str(model.optimizer) + '\n\n')

def save_summary(results_file, scenario, model, best_model, train_loader, dev_loader, test_loader, verbose=1):
    loaders = [('test',test_loader), ('dev',dev_loader), ('train',train_loader)]
    metrics = ['accuracy','precision','recall','f1']
    all_results = []

    for settype, loader in loaders:
        results = best_model.evaluate_metrics(loader, metrics)
        all_results.append((settype,results))
    
    with open(results_file, 'a') as outfile:
        prefix = hash_texto(str(scenario))
        outfile.write(str(scenario) + '\t')

        last_train_acc = model.train_history['train_acc'][-1]
        last_train_loss = model.train_history['train_loss'][-1]
        last_train_metrics = f'{last_train_acc:0.4f}\t{last_train_loss:0.4f}\t'
        outfile.write(last_train_metrics)

        for i, metric in enumerate(metrics):
            for settype, results in all_results:
                multiplier = 100 if metric == 'accuracy' else 1
                to_write = f'{results[i]*multiplier:02.4f}' + '\t'
                outfile.write(to_write)
                if verbose >= 1 and metric == 'accuracy':
                    to_report = f'{settype}_{metric}:{results[i]*multiplier:02.2f} '
                    sys.stdout.write(to_report)
        if verbose >= 1:
            print()
        outfile.write(prefix + '\n')

        


def save_model(model_path, scenario, model):
    outname = hash_texto(str(scenario)) + '.pkl'
    with open(model_path + outname, 'wb') as outfile:
        pickle.dump(model, outfile)

def load_model(model_path, scenario):
    filename = hash_texto(str(scenario)) + '.pkl'
    with open(model_path + filename, 'rb') as infile:
        model = pickle.load(infile)
    return model

def pad_collate(batch):
    batch.sort(key=lambda b: len(b[0]), reverse=True)
    X_seq, Y_seq = zip(*batch)
    X_seq = [torch.tensor(X) for X in X_seq]
    lengths = torch.tensor([len(X) for X in X_seq])
    X = torch.nn.utils.rnn.pad_sequence(X_seq)
    Y = torch.FloatTensor(Y_seq).view(-1,1)
    return (X, Y, lengths)

class MVSDataLoaderFactory():

    def __init__(self, batch_size=32, limit_vectors=None, dev_split=0.1, max_sequence_len=None):
        self.max_sequence_len = max_sequence_len
        self.dev_split = dev_split
        self.batch_size = batch_size
        MVSDataset.limit_vectors = limit_vectors

    def __data_set_from_label(self, label, set_type):
        X_file = config_data[label][set_type]['X']
        Y_file = config_data[label][set_type]['Y']
        language = config_data[label]['language']
        language_file = vector_files[language]
        dataset = MVSDataset(X_file, Y_file, language, language_file)
        return dataset

    def data_loaders_for_eval(self, labels=None, set_type='test'):
        list_of_data_loaders = []
        for label in labels:
            dataset = self.__data_set_from_label(label, set_type)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)
            list_of_data_loaders.append(data_loader)
        return list_of_data_loaders


    def data_loaders_from_scenario(self, scenario={'train':['es'], 'test':['es']}):
       
        list_of_train_datasets = []
        list_of_dev_datasets = []

        if 'dev' not in scenario:
        # if there is no dev set defined, we split the train datasets into train-dev datasets
        # considering the test set labels to make the split.    
            dev_labels = set(scenario['train']) & set(scenario['test'])
            if dev_labels == set():
                dev_labels = set(scenario['train'])
            total_size = 0
            to_dev_size = 0
            list_of_train_dev_datasets = []

            for label in scenario['train']:
                dataset = self.__data_set_from_label(label, 'train')
                list_of_train_dev_datasets.append(dataset)
                total_size += len(dataset)
                if label in dev_labels:
                    to_dev_size += len(dataset)

            corrected_dev_split = min(self.dev_split * total_size / to_dev_size, 1)

            for label, dataset in zip(scenario['train'], list_of_train_dev_datasets):
                if label not in dev_labels:
                    list_of_train_datasets.append(dataset)
                else: # we need to split
                    dev_size = int(len(dataset) * corrected_dev_split)
                    train_size = len(dataset) - dev_size
                    (to_dev, to_train) = random_split(dataset, [dev_size, train_size])
                    list_of_train_datasets.append(to_train)
                    list_of_dev_datasets.append(to_dev)
        else:
            for key, list_of_datasets in [('train', list_of_train_datasets), ('dev', list_of_dev_datasets)]:
                for label in scenario[key]:
                    dataset = self.__data_set_from_label(label, key)
                    list_of_datasets.append(dataset)

        train_dataset = ConcatDataset(list_of_train_datasets)
        dev_dataset = ConcatDataset(list_of_dev_datasets)

        list_of_test_datasets = []
        for label in scenario['test']:
            dataset = self.__data_set_from_label(label, 'test')
            list_of_test_datasets.append(dataset)

        test_dataset = ConcatDataset(list_of_test_datasets)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)

        return train_loader, dev_loader, test_loader


class MVSDataset(Dataset):
    ''' Multilingual Vector Sequence Dataset '''
    __multi_ling_vecs = {}
    __multi_ling_files = {}
    __vector_size = None
    limit_vectors = None
    __options_for_feminazi = {
        'es': ['mujer', 'feminista', 'chica', ''], 
        'en': ['woman', 'feminist', 'girl', ''],
        'it': ['ragazza', 'femminista', 'donna', '']
    }
    __options_for_feminazis = {
        'es': ['mujeres', 'feministas', 'chicas', ''],
        'en': ['women', 'feminists', 'girls', ''],
        'it': ['ragazze','femministe', 'donne', '']
    }

    def __init__(self, X_file, Y_file, language, 
        language_file=None, max_sequence_len=None, verbose=True):

        self.verbose = verbose
        self.raw_X = self.__load_file(X_file)
        self.raw_Y = self.__load_file(Y_file, transformation=int)
        assert len(self.raw_X) == len(self.raw_Y), 'X and Y data should have the same length.'
        self.language = language
        MVSDataset.load_vectors(language, language_file, verbose)
        self.max_sequence_len = max_sequence_len
        self.vectors = MVSDataset.__multi_ling_vecs[self.language]
        self.vector_size = MVSDataset.__vector_size

    def __len__(self):
        return len(self.raw_X)

    def __process_particular_words(self, word):
        if word == 'feminazi':
            word = np.random.choice(MVSDataset.__options_for_feminazi[self.language])
        elif word == 'feminazis':
            word = np.random.choice(MVSDataset.__options_for_feminazis[self.language])
        return word

    def __getitem__(self, idx):
        x, y = self.raw_X[idx], self.raw_Y[idx]
        x = x[:self.max_sequence_len]
        vec_list = []
        for token in x:
            token = self.__process_particular_words(token)
            if token in self.vectors:
                vec_list.append(torch.from_numpy(self.vectors[token]))
            else:
                # TODO: decide if zero is the best for OOV
                vec_list.append(torch.zeros(self.vector_size))
        x = torch.stack(vec_list)
        return (x,y)

    def __load_file(self, file_name, separator='\t', transformation=None):
        out = []
        with open(file_name) as file:
            for line in file:
                line = line[:-1]
                data = line.split(separator)
                if transformation:
                    data = [transformation(d) for d in data]
                out.append(data)
        return out

    @classmethod
    def load_vectors(cls, language, language_file, verbose=True):
        if language in cls.__multi_ling_vecs:
            if language_file:
                assert cls.__multi_ling_files[language] == language_file, 'Trying to load two different vector files for the same language.'
        else:
            if not language_file:
                err_msg = f'A file for language \'{language}\' should be loaded with '\
                    + 'MVSDataset.load_vectors or provided when initializing a dataset instance.' 
                raise ValueError(err_msg)
            else:
                if verbose:
                    print(f'Loading {cls.limit_vectors} vectors for language \'{language}\'.')
                cls.__multi_ling_files[language] = language_file 
                cls.__multi_ling_vecs[language] = KeyedVectors.load_word2vec_format(language_file, binary=False, limit=cls.limit_vectors)
                vector_size = cls.__multi_ling_vecs[language].vector_size
                if cls.__vector_size:
                    assert cls.__vector_size == vector_size, 'All vectors should be of the same size.'
                else:
                    cls.__vector_size = vector_size

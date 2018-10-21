import torch
import numpy as np
import hashlib
from torch.utils.data import Dataset, ConcatDataset, random_split, DataLoader
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from config_data import config_data, vector_files

sklearn_options = {
    'accuracy': accuracy_score,
    'f1': f1_score,
    'recall': recall_score,
    'precision': precision_score
}

def reset_all_seeds(RANDOM_SEED=33):
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


def score(Y_pred_prob, Y, sklearn_metric=accuracy_score):
    if type(sklearn_metric) == str:
        sklearn_metric = sklearn_options[sklearn_metric]
    Y_pred = (Y_pred_prob.float() >= 0.5).float()
    Y_pred = Y_pred.view(-1).cpu().numpy()
    Y = Y.view(-1).cpu().numpy()
    return sklearn_metric(Y, Y_pred)

def report(Y_pred_prob, Y):
    Y_pred = (Y_pred_prob >= 0.5).float()
    Y_pred = Y_pred.view(-1).numpy()
    Y = Y.view(-1).numpy()
    target_names = ['class_0', 'class_1']
    return classification_report(Y, Y_pred, target_names=target_names) 

def hash_texto(texto):
    out = hashlib.sha224(texto.encode('utf-8')).hexdigest()
    return str(out)

def load_scenarios(scenarios_file):
    with open(scenarios_file) as file:
        scenarios = []
        for line in file:
            scenario = eval(line[:-1])
            scenarios.append(scenario)
    return scenarios

def save_history(history_path, scenario, model):
    outname = hash_texto(str(scenario))[:15] + '.hist'
    with open(history_path + outname, 'w') as outfile:
        outfile.write(str(scenario) + '\n')
        outfile.write(str(model.history_output) + '\n')

def pad_collate(batch):
    batch.sort(key=lambda b: len(b[0]), reverse=True)
    X_seq, Y_seq = zip(*batch)
    X_seq = [torch.tensor(X) for X in X_seq]
    lengths = [len(X) for X in X_seq]
    X = torch.nn.utils.rnn.pad_sequence(X_seq)
    Y = torch.FloatTensor(Y_seq).view(-1,1)
    return (X, Y, lengths)


class MVSDataLoaderFactory():

    def __init__(self, batch_size=32, dev_split=0.1, max_sequence_len=None):
        self.max_sequence_len = max_sequence_len
        self.dev_split = dev_split
        self.batch_size = batch_size

    def __data_set_from_label(self, label, set_type):
        X_file = config_data[label][set_type]['X']
        Y_file = config_data[label][set_type]['Y']
        language = config_data[label]['language']
        language_file = vector_files[language]
        dataset = MVSDataset(X_file, Y_file, language, language_file)
        return dataset

    # TODO: dev split should be consistent with the test set (dev and test should have similar distributions!)
    def data_loaders_from_scenario(self, scenario={'train':['es'], 'test':'es'}):
        list_of_train_datasets = []
        for label in scenario['train']:
            dataset = self.__data_set_from_label(label, 'train')
            list_of_train_datasets.append(dataset)
        train_dev_dataset = ConcatDataset(list_of_train_datasets)
        dev_size = int(len(train_dev_dataset) * self.dev_split)
        train_size = len(train_dev_dataset) - dev_size
        (dev_dataset, train_dataset) = random_split(train_dev_dataset, [dev_size, train_size])
        test_dataset = self.__data_set_from_label(scenario['test'], 'test')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)

        return train_loader, dev_loader, test_loader


class MVSDataset(Dataset):
    ''' Multilingual Vector Sequence Dataset '''
    __multi_ling_vecs = {}
    __multi_ling_files = {}
    __vector_size = None

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

    def __getitem__(self, idx):
        x, y = self.raw_X[idx], self.raw_Y[idx]
        x = x[:self.max_sequence_len]
        vec_list = []
        for token in x:
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
                    print(f'Loading vectors for language \'{language}\'.')
                cls.__multi_ling_files[language] = language_file 
                cls.__multi_ling_vecs[language] = KeyedVectors.load_word2vec_format(language_file, binary=False)
                vector_size = cls.__multi_ling_vecs[language].vector_size
                if cls.__vector_size:
                    assert cls.__vector_size == vector_size, 'All vectors should be of the same size.'
                else:
                    cls.__vector_size = vector_size

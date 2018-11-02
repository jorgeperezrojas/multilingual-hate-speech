import numpy as np
import torch
import sys
import argparse
from hs_models import HS_Model
from torch.utils.data import DataLoader, ConcatDataset
from utils import reset_all_seeds, MVSDataLoaderFactory, get_hyperparameter_options, load_scenarios, save_history, load_model, save_summary, save_config
from config_data import vector_size, history_path, model_path, results_file, results_config_file, num_threads
import datetime
import pickle


RANDOM_SEED = 33
torch.set_num_threads(num_threads)

def main(eval_file, device, hyper_params_file, verbose):
    hps = get_hyperparameter_options(hyper_params_file)[0]
    batch_size = hps['batch_size']
    limit_vectors = hps['limit_vectors']
    dlf = MVSDataLoaderFactory(batch_size=batch_size, limit_vectors=limit_vectors)

    with open(eval_file) as infile:
        for line in infile:
            data = line[:-1].split(' ')
            base_lang = data[0]
            model_path = data[1]
            test_sets = data[2:]
            model = None

            data_loaders = dlf.data_loaders_for_eval(test_sets, 'test')
            with open(model_path, 'rb') as infile:
                model = pickle.load(infile)

            model.device = torch.device(device)

            print('lang',base_lang,'/ model', model_path)
            print(model.net)
            for loader, label in zip(data_loaders, test_sets):
                results = model.evaluate_metrics(loader)
                out = [str(x)[:6] for x in results]
                out = [label] + out
                print(' '.join(out))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('evalFile', metavar='F', type=str, 
        help='file with info for evaluating (models and data).')
    parser.add_argument('-d', '--device', metavar='D', default='cpu', 
        help='device to execute script (cpu or cuda:N, default=cpu)')
    parser.add_argument('-v', '--verbose', metavar='P', type=int, default=2, 
        help='verbosity level (default=2)')
    parser.add_argument('-y', '--hyperParams', metavar='Y', type=str, 
        help='file with specification for hyper parameter search.')

    args = parser.parse_args()
    main(args.evalFile, args.device, args.hyperParams, args.verbose)




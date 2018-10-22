import numpy as np
import torch
import sys
import argparse
from hs_models import HS_Model
from utils import MVSDataset, pad_collate
from torch.utils.data import DataLoader, ConcatDataset
from utils import reset_all_seeds, MVSDataLoaderFactory, load_scenarios, save_history, load_model, save_summary, save_config
from config_data import vector_size, history_path, model_path, results_file, batch_size, limit_vectors, config_file
import datetime


RANDOM_SEED = 33

def main(scenarios_file, device, epochs, patience, verbose):
    dlf = MVSDataLoaderFactory(batch_size=batch_size, limit_vectors=limit_vectors)
    scenarios = load_scenarios(scenarios_file)
    print('training in', device)

    # dumb model only to save specs
    dmodel = HS_Model(vector_size=vector_size, device=device, patience=patience)
    save_config(dmodel, config_file)

    for scenario in scenarios:
        reset_all_seeds(RANDOM_SEED)

        print('\nTraining scenario:',scenario)
        train_loader, dev_loader, test_loader = dlf.data_loaders_from_scenario(scenario)
        model = HS_Model(vector_size=vector_size, device=device, patience=patience,
            save_best=True, scenario=scenario, model_path=model_path)
        model.train(train_loader, dev_loader, epochs=epochs, verbose=verbose)
        best_model = load_model(model_path, scenario)
        save_summary(results_file, scenario, best_model, train_loader, dev_loader, test_loader, verbose=1)
        print('Finish training scenario:',scenario)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scenarios', metavar='S', type=str, 
        help='file with scenarios to train and test.')
    parser.add_argument('-d', '--device', metavar='D', default='cpu', 
        help='device to execute script (cpu or cuda:N, default=cpu)')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300, 
        help='number of epochs (default=300).')
    parser.add_argument('-y', '--hyperParams', metavar='Y', type=str, 
        help='file with specification for hyper parameter search.')
    parser.add_argument('-p', '--patience', metavar='P', type=int, default=30, 
        help='maximum number of repetitions allowed without improvement (default=30)')
    parser.add_argument('-v', '--verbose', metavar='P', type=int, default=2, 
        help='verbosity level (default=2)')
    args = parser.parse_args()
    main(args.scenarios, args.device, args.epochs, args.patience, args.verbose)




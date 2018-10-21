import numpy as np
import torch
import sys
import argparse
from hs_models import HS_Model
from utils import MVSDataset, pad_collate
from torch.utils.data import DataLoader, ConcatDataset
from utils import reset_all_seeds, MVSDataLoaderFactory, load_scenarios, save_history
from config_data import vector_size, history_path

RANDOM_SEED = 33

def main(scenarios_file, device, epochs, patience):
    dlf = MVSDataLoaderFactory()
    scenarios = load_scenarios(scenarios_file)
    print('training in', device)
    for scenario in scenarios:
        dlf.batch_size = 16
        reset_all_seeds(RANDOM_SEED)
        print('training scenario:',scenario)
        train_loader, dev_loader, test_loader = dlf.data_loaders_from_scenario(scenario)
        model = HS_Model(vector_size=vector_size, device=device, patience=patience)
        model.train(train_loader, dev_loader, epochs=epochs)
        test_loss, test_acc = model.evaluate(test_loader)
        save_history(history_path, scenario, model)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scenarios', metavar='S', type=str, 
        help='file with scenarios to train and test.')
    parser.add_argument('-d', '--device', metavar='D', default='cpu', 
        help='device to execute script (cpu or cuda:N)')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300, 
        help='number of epochs.')
    parser.add_argument('-y', '--hyperParams', metavar='Y', type=str, 
        help='file with specification for hyper parameter search.')
    parser.add_argument('-p', '--patience', metavar='P', type=int, default=30, 
        help='maximum number of repetitions allowed without improvement')
    args = parser.parse_args()
    main(args.scenarios, args.device, args.epochs, args.patience)




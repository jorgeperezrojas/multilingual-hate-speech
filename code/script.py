import numpy as np
import torch
import sys
import argparse
from hs_models import HS_Model
from utils import MVSDataset, pad_collate
from torch.utils.data import DataLoader, ConcatDataset
from utils import reset_all_seeds, MVSDataLoaderFactory, load_scenarios, save_history, load_model, save_summary
from config_data import vector_size, history_path, model_path, results_file
import datetime


RANDOM_SEED = 33

def main(scenarios_file, device, epochs, patience):
    dlf = MVSDataLoaderFactory()
    scenarios = load_scenarios(scenarios_file)
    print('training in', device)

    # dumb model only to save specs
    dmodel = HS_Model(vector_size=vector_size, device=device, patience=patience)
    with open(results_file, 'w') as outfile:
        outfile.write(str(datetime.datetime.now()) + '\n')
        outfile.write(str(dmodel.net) + '\n')
        outfile.write(str(dmodel.optimizer) + '\n')

    for scenario in scenarios:
        dlf.batch_size = 32
        reset_all_seeds(RANDOM_SEED)
        print('training scenario:',scenario)
        train_loader, dev_loader, test_loader = dlf.data_loaders_from_scenario(scenario)
        model = HS_Model(vector_size=vector_size, device=device, patience=patience,
            save_best=True, scenario=scenario, model_path=model_path)
        model.train(train_loader, dev_loader, epochs=epochs)
        save_history(history_path, scenario, model)

        # carga y evalua el mejor modelo
        best_model = load_model(model_path, scenario)
        test_loss, test_acc = best_model.evaluate(test_loader)
        
        print('finish scenario:',scenario)
        print(f'in best model: test_loss:{test_loss:02.4f}, test_acc:{test_acc*100:02.2f}%')
        save_summary(results_file, scenario, test_acc, test_loss, model)


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




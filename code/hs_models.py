import torch
import sys
import time
import numpy as np
from utils import score, report
from collections import defaultdict


class LSTM_HS(torch.nn.Module):

    def __init__(self,
        vector_size = 300,
        lstm_hidden_size = 128,
        lstm_layers = 3,
        bidirectional = True,
        lstm_dropout = 0.4,
        fc_hidden_size = 500,
        fc_dropout = 0.3
    ):
        super(LSTM_HS, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.directions = 2 if bidirectional else 1
        
        self.lstm = torch.nn.LSTM(
            input_size = vector_size,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_layers,
            dropout = lstm_dropout,
            bidirectional = bidirectional)
        self.fc_1 = torch.nn.Linear(self.directions * lstm_hidden_size, fc_hidden_size)
        self.fc_2 = torch.nn.Linear(fc_hidden_size, 1)
     

    def forward(self, X, lengths):
        # asume que el input es una sequencia con padding
        # asume time-step first
        X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths)
        _, (h, _) = self.lstm(X) 
        # aqui desarmo
        h = h.view(self.lstm_layers, self.directions, -1, self.lstm_hidden_size)  
        h = h[-1].transpose(0,1).contiguous()
        h = h.view(-1, self.directions * self.lstm_hidden_size)
        ##############
        h = torch.nn.functional.relu(self.fc_1(h))
        h = self.fc_2(h)
        return h


class HS_Model():

    def __init__(self, max_sequence_len=None, device='cpu', patience=None, **kwargs):
        self.device = torch.device(device)
        self.max_sequence_len = max_sequence_len
        self.net = LSTM_HS(**kwargs)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.detailed_train_history = defaultdict(list)
        self.train_history = defaultdict(list)
        self.best_dev_loss = {}
        self.best_dev_loss['value'] = 1 
        self.best_dev_acc = {}
        self.best_dev_acc['value'] = 0
        self.patience = patience
        self.epochs_not_improving = 0
        self.history_output = {
                'history': self.train_history,
                'details': self.detailed_train_history,
                'best_loss': self.best_dev_loss,
                'best_acc': self.best_dev_acc,
            }

    def train(self, train_loader, dev_loader, epochs=2, verbose=1):      
        batch_size = train_loader.batch_size
        iterations = len(train_loader)

        for epoch in range(epochs):
            running_loss = 0.0
            running_acc = 0.0
            epoch_init_time = time.time()

            for i, batch in enumerate(train_loader):

                self.net = self.net.to(self.device).train()
                self.optimizer.zero_grad()
                X, Y, lengths = batch
                X, Y = X.to(self.device), Y.to(self.device)
                Y_pred = self.net(X, lengths)
                loss = self.criterion(Y_pred, Y)
                loss.backward()
                self.optimizer.step()
                
                running_acc += score(Y_pred, Y)
                running_loss += loss.item()
                
                if verbose == 1:
                    self.__report_iteration(iterations, i, epoch_init_time, 
                                            epoch, running_loss, running_acc, verbose)
                self.__log_detailed_history(epoch,i,running_loss, running_acc)
                    
            if dev_loader:
                dev_loss, dev_acc = self.evaluate(dev_loader)
            else:
                dev_loss, dev_acc = None, None
            self.__report_epoch(epoch, epochs, epoch_init_time, iterations, 
                                running_loss, running_acc, dev_loss, dev_acc, verbose)
            self.__log_history(epoch, iterations, running_loss, running_acc, dev_loss, dev_acc)
            
            if self.patience and self.epochs_not_improving >= self.patience:
                print('\nEnough of not improving...')
                break

        return self.history_output


    def evaluate(self, data_loader):
        with torch.no_grad():
            running_loss = 0
            running_acc = 0
            for batch in data_loader:
                self.net = self.net.to(self.device).eval()
                X, Y, lengths = batch
                X, Y = X.to(self.device), Y.to(self.device)
                Y_pred = self.net(X, lengths)
                running_loss += self.criterion(Y_pred, Y).item()
                running_acc += score(Y_pred, Y)
            loss = running_loss/len(data_loader)
            acc = running_acc/len(data_loader)
        return loss, acc


    def __log_detailed_history(self, epoch,iteration,running_loss, running_acc):
        self.detailed_train_history['train_loss'].append(running_loss/(iteration+1))
        self.detailed_train_history['train_acc'].append(running_acc/(iteration+1))

    def __log_history(self, epoch, iterations, running_loss, running_acc, dev_loss, dev_acc):
        train_loss = running_loss/iterations
        train_acc = running_acc/iterations
        self.train_history['train_loss'].append(train_loss)
        self.train_history['train_acc'].append(train_acc)
        self.train_history['dev_loss'].append(dev_loss)
        self.train_history['dev_acc'].append(dev_acc)
        if dev_loss < self.best_dev_loss['value']:
            self.best_dev_loss['value'] = dev_loss
            self.best_dev_loss['epoch'] = epoch
        if dev_acc < self.best_dev_acc['value']:
            self.best_dev_acc['value'] = dev_acc
            self.best_dev_acc['epoch'] = epoch
        else:
            self.epochs_not_improving += 1


    def __report_iteration(self, iterations, i, epoch_init_time, epoch,
                        running_loss, running_acc, verbose):
        completion = int((i+1) / iterations * 100)
        ETA = (time.time() - epoch_init_time) / (i+1) * (iterations - i - 1)
        out_info_iter = \
            f'\rEpoch:{epoch+1}, ' +\
            f'progress:{completion:02.0f}%, ETA:{ETA:03.0f}s, ' +\
            f'loss:{running_loss/(i+1):02.4f}, ' +\
            f'accuracy:{running_acc/(i+1)*100:02.2f}%    '
        sys.stdout.write(out_info_iter)
    

    def __report_epoch(self, epoch, epochs, epoch_init_time, iterations, 
                       running_loss, running_acc, dev_loss, dev_acc, verbose):
        total_time = time.time() - epoch_init_time
        out_info_epoch = \
            f'Epoch:{epoch+1}/{epochs}, ' +\
            f'total time:{total_time:2.0f}s, ' +\
            f'time/update:{total_time/iterations:.2f}s, ' +\
            f'loss:{running_loss/iterations:02.4f}, ' +\
            f'accuracy: {running_acc/iterations*100:02.2f}%'
        if dev_loss and dev_acc:
            out_info_epoch += ', ' +\
                f'dev_loss:{dev_loss:02.4f}, ' +\
                f'dev_acc:{dev_acc*100:02.2f}% '
        if verbose == 1:
            print('\n' + out_info_epoch + '\n')
        elif verbose == 2:
            sys.stdout.write('\r' + out_info_epoch)
        else:
            pass

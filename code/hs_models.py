import torch
import sys
import time
import numpy as np
from utils import score, report, save_model
from collections import defaultdict
import pickle


class LSTM_HS(torch.nn.Module):

    def __init__(self,
        initial_avg,
        avg_size,
        input_dropout,
        lstm_hidden_size,
        lstm_layers,
        bidirectional,
        lstm_dropout,
        fc_hidden_size,
        fc_dropout,
        vector_size = 300,
        **kwargs
    ):
        super(LSTM_HS, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.directions = 2 if bidirectional else 1
        self.fc_hidden_size = fc_hidden_size

        self.inptu_dropout = torch.nn.Dropout(input_dropout)

        self.initial_avg = initial_avg
        if self.initial_avg:
            assert avg_size % 2 == 1, "Size of the average pooling should be an odd number."
            padding_size = int(avg_size/2)
            stride = 1
            self.avg_pool = torch.nn.AvgPool1d(avg_size,stride,padding_size)
        
        self.lstm = torch.nn.LSTM(
            input_size = vector_size,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_layers,
            dropout = lstm_dropout,
            bidirectional = bidirectional)
        
        self.fc_1_dropout = torch.nn.Dropout(fc_dropout)
        self.fc_1 = torch.nn.Linear(self.directions * lstm_hidden_size, fc_hidden_size)
        self.fc_2_dropout = torch.nn.Dropout(fc_dropout)
        self.fc_2 = torch.nn.Linear(fc_hidden_size, 1)
     

    def forward(self, X, lengths):
        # asume que el input es una sequencia con padding
        # asume time-step first (L,N,C)
        X = self.inptu_dropout(X)

        if self.initial_avg:
            # rearrange to apply 1d average (L,N,C) --> (N,C,L)
            X = X.transpose(0,1).transpose(1,2)
            X = self.avg_pool(X)
            # rearrange again for the input of the recurrent layer (N,C,L) --> (L,N,C)
            X = X.transpose(2,1).transpose(1,0)
            
        X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths)
        _, (h, _) = self.lstm(X) 
        # aqui desarmo
        h = h.view(self.lstm_layers, self.directions, -1, self.lstm_hidden_size)  
        h = h[-1].transpose(0,1).contiguous()
        h = h.view(-1, self.directions * self.lstm_hidden_size)
        ##############
        h = self.fc_1_dropout(h)
        h = self.fc_1(h)
        if self.fc_hidden_size > 1:
            h = torch.nn.functional.relu(h)
            h = self.fc_2(h)
        return h


class HS_Model():
    def __init__(self, max_sequence_len=None, device='cpu', patience=None, 
            save_best=False, scenario=None, model_path=None, optimizer='sgd',
            lr=0.01, momentum=0.9, weight_decay=0, **kwargs):    
        self.device = torch.device(device)
        self.max_sequence_len = max_sequence_len

        # net
        self.net = LSTM_HS(**kwargs)

        # optimizer
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)        
        
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

        assert (not save_best) or (scenario and model_path), 'For save_best True, scenario and model_path should be defined'
        self.save_best = save_best
        self.scenario = scenario
        self.model_path = model_path

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
                best_dev_loss = self.best_dev_loss['value']
                best_dev_acc = self.best_dev_acc['value']
                final_msg = 'Enough of not improving...' +\
                    f'best_dev_loss:{best_dev_loss:02.4f}, ' +\
                    f'best_dev_acc:{best_dev_acc*100:02.2f}%'
                print(final_msg)
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

    def evaluate_metrics(self, data_loader, metrics=['accuracy','precision','recall','f1']):
        out_metrics = []
        Ys, Ypreds = [], []
        with torch.no_grad():
            for batch in data_loader:
                self.net = self.net.to(self.device).eval()
                X, Y, lengths = batch
                X, Y = X.to(self.device), Y.to(self.device)
                Y_pred = self.net(X, lengths)
                Ys.append(Y)
                Ypreds.append(Y_pred)
            Y = torch.cat(Ys)
            Y_pred = torch.cat(Ypreds)
            for metric in metrics:
                out_metrics.append(score(Y_pred, Y, metric))
        return out_metrics



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
        if dev_acc > self.best_dev_acc['value']:
            self.best_dev_acc['value'] = dev_acc
            self.best_dev_acc['epoch'] = epoch
            self.epochs_not_improving = 0
            if self.save_best:
                save_model(self.model_path, self.scenario, self)
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
            f'accuracy:{running_acc/iterations*100:02.2f}%'
        if dev_loss and dev_acc:
            out_info_epoch += ', ' +\
                f'dev_loss:{dev_loss:02.4f}, ' +\
                f'dev_acc:{dev_acc*100:02.2f}% '
        if verbose == 1:
            print('\n' + out_info_epoch + '\n')
        elif verbose == 2:
            print(out_info_epoch)
        else:
            pass

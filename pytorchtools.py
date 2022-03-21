import numpy as np
import torch
import os

class EarlyStopping:
    def __init__(self, patience = 5, verbose = False, delta = 0, path ='checkpoint'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.ealry_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        if not os.path.isdir(path):
            os.mkdir(path)

        self.path = os.path.join(path, 'checkpoint.pt')
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.ealry_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        감소할 때마다 모델 저장
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
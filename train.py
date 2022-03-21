import time
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import argparse
import os
from tqdm import tqdm

from pytorchtools import EarlyStopping
from model_mobilenet_v2 import save_model, init_mobilenet
from train_valid_loader import get_train_valid_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--dataset', type=str, required = True)
args = parser.parse_args()

def set_randomseed(num = 0):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    np.random.seed(num)
    random.seed(num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_earlystop(model, train_loader, valid_loader, criterion, optimizer, scheduler, patience = 5, num_epochs = 25):
    """
    -----return-----
    """
    since = time.time()
    train_losses = []
    valid_losses = []

    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience = patience, verbose = True)

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#       print('-' * 10)

        #train phase
        model.train()
        for data, target in tqdm(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        
        #valid phase
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)

                valid_losses.append(loss.item())


        #에포크당 평균 loss
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(num_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                f'train_loss: {train_loss:.5f} ' +
                f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.ealry_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(early_stopping.path))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, avg_train_losses, avg_valid_losses


def main():
    DATA_DIR = 'data/train'
    MODEL_DIR = 'model'
    
    try:
        train_loader, valid_loader = get_train_valid_loader(os.path.join(DATA_DIR, args.dataset), args.batch_size, augment=True)
    except:
        print('failed to load dataset')
        exit()
    # model 초기화
    set_randomseed(0)
    model = init_mobilenet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier[1].parameters(), lr = args.lr, momentum = 0.9)

    # decay_gamma = 0.001
    # decay_step = 3
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_gamma)

    #훈련
    model, train_losses, valid_losses = train_earlystop(model, train_loader, valid_loader, criterion, optimizer, num_epochs= args.epochs)

    #베스트 모델 저장
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    save_model(model, MODEL_DIR, f'_{args.dataset}_lr_{args.lr}_batch_{args.batch_size}_epochs_{args.epochs}_valloss{np.round(min(valid_losses), 4)}')
    
    if device != 'cpu':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DatasetInstance, get_sudoku_dataset
from model import sudoku_lpconv, SudokuCNN
from metrics import eval_num_accs, eval_sudokus

def train(device, train_loader, model, criterion, optimizer, verbose=False, debug=False):
    loss_sum = 0
    accs = []
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss = criterion(outputs , targets)

        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if 'log2p' in name:
                param.data.clamp_(min=1) # for numerical stability

        optimizer.step()

        loss_sum += loss.item()
        preds = torch.argmax(outputs,dim=1)

        num_accs = eval_num_accs(preds.detach().cpu(), targets.detach().cpu())
        acc = eval_sudokus(preds.detach().cpu())
        acc['number'] = num_accs
        accs.append(acc)

        avg_train_loss = loss_sum/(i+1)
        sudoku_train_accs = pd.concat(accs)

        if verbose and (i%100==0):
            avg_acc = ' '.join(f'{sudoku_train_accs.mean()}'.split()[:-2])
            print(f'[{i}/{len(train_loader)}] Avg Loss: {avg_train_loss:.4f} Avg Acc: {avg_acc}', flush=True)

        if debug:
            break

    return avg_train_loss, sudoku_train_accs

def test(device, test_loader, model, criterion, verbose=False, debug=False):
    with torch.no_grad():
        loss_sum = 0
        accs = []
        model.eval()
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs , targets)

            loss_sum += loss.item()
            preds = torch.argmax(outputs,dim=1)

            num_accs = eval_num_accs(preds.detach().cpu(), targets.detach().cpu())
            acc = eval_sudokus(preds.detach().cpu())
            acc['number'] = num_accs
            accs.append(acc)

            avg_test_loss = loss_sum/(i+1)
            sudoku_test_accs = pd.concat(accs)

            if verbose and (i%100==0):
                avg_acc = ' '.join(f'{sudoku_test_accs.mean()}'.split()[:-2])
                print(f'[{i}/{len(test_loader)}] Avg Loss: {avg_test_loss:.4f} Avg Acc: {avg_acc}', flush=True)
            
            if debug:
                break

    return avg_test_loss, sudoku_test_accs

def bechmark(model=SudokuCNN(), save_dir='../output_dir/sudoku', data_path='../sudoku.csv', batch_size=100, lr=3e-4, epochs=10, seed=0, verbose=False, debug=False):
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset = get_sudoku_dataset(data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_loss = float('inf')  # initialize to a very high value

    iter_train_accs = []
    iter_test_accs = []
    avg_train_accs = []
    avg_test_accs = []
    avg_train_losses = []
    avg_test_losses = []
    for epoch in range(epochs):
        avg_train_loss, sudoku_train_accs = train(device, train_loader, model, criterion, optimizer, verbose=verbose, debug=debug)
        avg_test_loss, sudoku_test_accs = test(device, test_loader, model, criterion, verbose=verbose, debug=debug)
        iter_train_accs.append( sudoku_train_accs )
        iter_test_accs.append( sudoku_test_accs )
        avg_train_accs.append( sudoku_train_accs.mean() )
        avg_test_accs.append( sudoku_test_accs.mean() )
        avg_train_losses.append( avg_train_loss )
        avg_test_losses.append( avg_test_loss )


        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            checkpoint_path = os.path.join(save_dir, 'sudoku_cnn_checkpoint.pt')
            torch.save(model.state_dict(), checkpoint_path)

        avg_train_acc = ' '.join(f'{sudoku_train_accs.mean()}'.split()[:-2])
        avg_test_acc = ' '.join(f'{sudoku_test_accs.mean()}'.split()[:-2])
        print(f'[Epochs: {epoch}/{epochs}] Avg Train Loss: {avg_train_loss:.4f} Avg Train Acc: {avg_train_acc}', flush=True)
        print(f'[Epochs: {epoch}/{epochs}] Avg Test Loss: {avg_test_loss:.4f} Avg Test Acc: {avg_test_acc}', flush=True)

        pd.concat(iter_train_accs).reset_index(drop=True).to_csv( os.path.join(save_dir, 'iter_train_accs.csv'))
        pd.concat(iter_test_accs).reset_index(drop=True).to_csv( os.path.join(save_dir, 'iter_test_accs.csv'))
        pd.concat(avg_train_accs, axis=1).T.to_csv( os.path.join(save_dir, 'avg_train_accs.csv'))
        pd.concat(avg_test_accs, axis=1).T.to_csv( os.path.join(save_dir, 'avg_test_accs.csv'))
        pd.DataFrame(dict(train_loss=avg_train_losses, test_loss=avg_test_losses)).to_csv( os.path.join(save_dir, 'loss.csv') )

if __name__ == "__main__":
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--log2p', '-p', type=float, default=-1)
    parser.add_argument('--num-hidden', '-h', type=int, default=512)
    parser.add_argument('--num-layers', '-l', type=int, default=15)
    parser.add_argument('--save-dir', '-sd', type=str, default='../output_dir/sudoku')
    parser.add_argument('--data-path', '-dp', type=str, default="../sudoku.csv")
    parser.add_argument('--batch-size', '-bs', type=int, default=100)
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--lr', '-lr', type=float, default=1e-4)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--lpconvert', action='store_true')
    parser.add_argument('--lpfrozen', action='store_true')

    args = parser.parse_args()
    if args.log2p == -1:
        args.log2p = None
    if args.lpconvert is False:
        args.log2p = 'base'
    print(vars(args))

    model = sudoku_lpconv(num_hidden=args.num_hidden, num_layers=args.num_layers, log2p=args.log2p, lpconvert=args.lpconvert, learnable=(not args.lpfrozen))
    save_dir = os.path.join(args.save_dir, f'num_layers={args.num_layers}', f'num_hidden={args.num_hidden}', f'log2p={args.log2p}', f'seed={args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(args, os.path.join(save_dir, 'args.pt'))
    bechmark(
        model=model, 
        save_dir=save_dir, 
        data_path=args.data_path, 
        batch_size=args.batch_size, 
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        verbose=args.verbose, 
        debug=args.debug)
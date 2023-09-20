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
from main import train, test, benchmark

def list_experiments(directory):
    experiments = {}
    for dirpath, dirnames, filenames in os.walk(directory):
        if 'args.pt' in os.listdir(dirpath):
            experiments[dirpath] = {}
            experiments[dirpath]['args'] = torch.load(os.path.join(dirpath, 'args.pt'), map_location='cpu')
            experiments[dirpath]['checkpoint']=torch.load(os.path.join(dirpath, 'sudoku_cnn_checkpoint.pt'), map_location='cpu')
    print(experiments.keys(), flush=True)
    return experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('-d', '--directory', type=str, default=r'/mnt/lustre/ibs/jeakwon/project2023/lpconv/output_dir/sudoku/num_layers=10/num_hidden=256/log2p=None/', help='Directory path')
    parser.add_argument('--new-log2p', type=int, default=1)
    args = parser.parse_args()

    print(vars(args), flush=True)

    experiments = list_experiments(args.directory)
    print(experiments.keys(), flush=True)

    for dirpath in experiments.keys():
        exp_args = experiments[dirpath]['args']
        checkpoint = experiments[dirpath]['checkpoint']
        old_model = sudoku_lpconv(num_hidden=exp_args.num_hidden, num_layers=exp_args.num_layers, log2p=exp_args.log2p, lpconvert=exp_args.lpconvert, learnable=(not exp_args.lpfrozen))
        old_model.load_state_dict(checkpoint)

        new_model = sudoku_lpconv(num_hidden=exp_args.num_hidden, num_layers=exp_args.num_layers, log2p=args.new_log2p, lpconvert=exp_args.lpconvert, learnable=(not exp_args.lpfrozen))
        
        for p in old_model.named_parameters():
            print(p)
        for p in new_model.named_parameters():
            print(p)


    # parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--log2p', '-p', type=float, default=-1)
    # parser.add_argument('--num-hidden', '-h', type=int, default=512)
    # parser.add_argument('--num-layers', '-l', type=int, default=15)
    # parser.add_argument('--save-dir', '-sd', type=str, default='../output_dir/sudoku')
    # parser.add_argument('--data-path', '-dp', type=str, default="../sudoku.csv")
    # parser.add_argument('--batch-size', '-bs', type=int, default=100)
    # parser.add_argument('--epochs', '-e', type=int, default=20)
    # parser.add_argument('--lr', '-lr', type=float, default=1e-4)
    # parser.add_argument('--verbose', action='store_true')
    # parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--lpconvert', action='store_true')
    # parser.add_argument('--lpfrozen', action='store_true')

    # args = parser.parse_args()
    # if args.log2p == -1:
    #     args.log2p = None
    # if args.lpconvert is False:
    #     args.log2p = 'base'
    # print(vars(args))

    # model = sudoku_lpconv(num_hidden=args.num_hidden, num_layers=args.num_layers, log2p=args.log2p, lpconvert=args.lpconvert, learnable=(not args.lpfrozen))
    # save_dir = os.path.join(args.save_dir, f'num_layers={args.num_layers}', f'num_hidden={args.num_hidden}', f'log2p={args.log2p}', f'seed={args.seed}')
    # os.makedirs(save_dir, exist_ok=True)
    # torch.save(args, os.path.join(save_dir, 'args.pt'))
    # bechmark(
    #     model=model, 
    #     save_dir=save_dir, 
    #     data_path=args.data_path, 
    #     batch_size=args.batch_size, 
    #     lr=args.lr,
    #     epochs=args.epochs,
    #     seed=args.seed,
    #     verbose=args.verbose, 
    #     debug=args.debug)
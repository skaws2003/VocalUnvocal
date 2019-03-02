import torch
import torchvision
import os
import argparse
from vocal_data import get_vocaldata
from vocal_unvocal_net import VocalUnvocal

# Argument parser settings
parser = argparse.ArgumentParser()

# Network options
parser.add_argument('--hidden_size',type=int,   default=64,     help='Hidden layer size for GRUcell')
parser.add_argument('--num_layers', type=int,   default=1,      help='Number of layers for GRUcell')

# Dataset options
parser.add_argument('--data_root',  type=str,                   help='Location to Dataset root')
parser.add_argument('--max_length', type=int,   default=20000,  help='Maximum length of mfcc features to analyze')
parser.add_argument('--batch_size', type=int,   default=1,      help='Batch size used for training')

# Training Options
parser.add_argument('--init_lr',    type=float, default=1e-3,   help='Initial learning rate')
parser.add_argument('--debug',      type=bool,  default=False,  action='store_True',    help='Run in debug mode(num_worker=0)')
parser.add_argument('--dropout',    type=float, default=0.5,    help='Dropout rate used for training')

args = parser.parse_args()


def train():
    
    raise NotImplementedError("train() is not implemented yet.")


def eval():
    raise NotImplementedError("eval() is not implemented yet.")


def main():
    model = VocalUnvocal(input_size=13, hidden_size=args.hidden_size, n_layers=args.num_layers, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)



if __name__=='__main__':
    main()
import torch
from torch.optim import lr_scheduler
import torchvision
import os
import argparse
from vocal_data import get_vocaldata
from models.vocal_unvocal_net import VocalUnvocalNet
import numpy as np

# Argument parser settings
parser = argparse.ArgumentParser()

# Network options
parser.add_argument('--hidden_size',type=int,   default=64,     help='Hidden layer size for GRUcell')
parser.add_argument('--num_layers', type=int,   default=1,      help='Number of layers for GRUcell')

# Dataset options
parser.add_argument('--data_root',  type=str,                   help='Location to Dataset root')
parser.add_argument('--max_length', type=int,   default=10000,  help='Maximum length of mfcc features to analyze')
parser.add_argument('--batch_size', type=int,   default=1,      help='Batch size used for training')

# Training Options
parser.add_argument('--init_lr',    type=float, default=1e-3,   help='Initial learning rate')
parser.add_argument('--debug',      action='store_true',        help='Run in debug mode(num_worker=0)')
parser.add_argument('--dropout',    type=float, default=0.5,    help='Dropout rate used for training')
parser.add_argument('--cuda',       type=int,   default=1,      help='cuda device number.')
parser.add_argument('--epochs',     type=int,   default=400,    help='training epochs')

parser.set_defaults(debug=False)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')


def train(model,optimizer,scheduler,train_loader,val_loader,criterion,epochs):

    epoch_losses_train = []
    epoch_losses_eval = []
    best_loss = None
    best_model = None
    for epoch in range(epochs):
        print("epoch %d started."%(epoch))
        # Train
        model.train()
        losses = []
        acc_list_train = []
        for i,(data,target,lengths) in enumerate(train_loader):
            # forward path
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            lengths = lengths.to(device)
            result = model(data,lengths)
            loss = criterion(result,target)
            losses.append(loss.item())
            # backward path
            loss.backward()
            optimizer.step()

            # Accuracy
            batch_pred = np.argmax(result.detach().cpu().numpy(),axis=1)
            accu = np.mean(batch_pred == target.detach().cpu().numpy())
            acc_list_train.append(accu.item())
            if i % 50 == 0:
                print("%d/%d"%(i,len(train_loader)))
                
        print("Training done. Training acc: %lf"%(np.mean(acc_list_train)))

        # model update
        epoch_loss = np.mean(losses)
        if best_loss is not None:
            if epoch_loss < best_loss:      # if better
                best_loss = epoch_loss
                best_model = model.state_dict()
        else:
            best_loss = epoch_loss
            best_model = model.state_dict()
        print("EPOCH %d: train loss = %.05f"%(epoch,epoch_loss))
        epoch_losses_train.append(epoch_loss)

        # Eval
        model.eval()
        losses = []
        total_num = 0
        total_acc = 0
        with torch.no_grad():
            for i,(data,target,lengths) in enumerate(val_loader):
                data = data.to(device)
                target = target.to(device)
                lengths = lengths.to(device)
                result = model(data,lengths)
                loss = criterion(result,target)
                losses.append(loss.item())
                # accu
                batch_pred = np.argmax(result.detach().cpu().numpy(),axis=1)
                accu = (batch_pred == target.detach().cpu().numpy())
                total_num += len(target)
                total_acc += sum(accu)
            epoch_loss = np.mean(losses)
            scheduler.step(epoch_loss)
            print("EPOCH %d: validation loss = %.05f"%(epoch,epoch_loss))
            print("Validation accuracy = %.05f"%(total_acc/total_num))
            epoch_losses_eval.append(epoch_loss)

    # Load best model
    model.load_state_dict(best_model)


def main():
    # Load models, dataset, dataloaders
    model = VocalUnvocalNet(input_size=13, hidden_size=args.hidden_size, n_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.33,patience=8)
    train_set, val_set = get_vocaldata(root=args.data_root, length=args.max_length)
    if args.debug:
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    criterion = torch.nn.CrossEntropyLoss()

    # Now train!
    train(model,optimizer,scheduler,train_loader,val_loader,criterion,args.epochs)


if __name__=='__main__':
    main()
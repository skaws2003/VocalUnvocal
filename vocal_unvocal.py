"""
a network for VocalUnvocal project.
"""

import torch.nn as nn
import torch
import torchvision

class VocalUnvocal(nn.modules):
    def __init__(self,input_size,hidden_size,n_layers=1,dropout=0):
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=self.n_layers, dropout = (0 if n_layers==1 else self.dropout),
            bidirectional = True
        )

    def forward(self, mfcc_features,lengths,hidden=None):
        packed = nn.utils.rnn.pack_padded_sequence(mfcc_features,lengths)
        outputs, hidden = self.gru(packed,hidden)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:,:,self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs, hidden
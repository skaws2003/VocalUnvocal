"""
a network for VocalUnvocal project.
"""

import torch.nn as nn
import torch
import torchvision

class VocalUnvocal_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(VocalUnvocal_GRU,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=self.n_layers, dropout = (0 if n_layers==1 else self.dropout),
            bidirectional = True,
            batch_first=True
        )

    def forward(self, inputs, lengths, hidden=None):
        packed_data = torch.nn.utils.rnn.pack_padded_sequence(inputs,lengths,batch_first=True)
        outputs, hidden = self.gru(packed_data)
        outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs, hidden

class VocalUnvocalNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(VocalUnvocalNet,self).__init__()
        self.vocalnet = VocalUnvocal_GRU(input_size=input_size,hidden_size=hidden_size,n_layers=n_layers,dropout=dropout)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,padding=2),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=5,padding=2)
        )
        self.after_conv = nn.Conv2d(in_channels=32,out_channels=2,kernel_size=3,padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        

    def forward(self, packed: nn.utils.rnn.PackedSequence, hidden):
        gru_output,_ = self.vocalnet(packed,hidden)
        to_conv = gru_output.unsqueeze(1)
        conv3s = self.conv3(to_conv)
        conv5s = self.conv5(to_conv)
        conv_cat = torch.cat([conv3s,conv5s],dim=1)
        conved = self.after_conv(conv_cat)
        pooled = self.global_pool(conved)
        final = pooled.squeeze(3).squeeze(2)

        return final
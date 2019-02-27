"""
a dataset for VocalUnvocal project.
"""

import torch
import os
import numpy as np

class VocalData(torch.utils.data.Dataset):
    """
    A pytorch dataset class for vocal dataset.
    """
    def __init__(self,root,length=20000):
        """
        :param root: directory to the dataset.
        :param length: length of mfcc features to look at
        """
        self.root = root
        self.length = length
        on_vocal = os.listdir(os.path.join(self.root,'on'))
        off_vocal = os.listdir(os.path.join(self.root,'off'))
        
        self.data = []
        for f in on_vocal:
            self.data.append((f,0))
        for f in off_vocal:
            self.data.append((f,1))


    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        filename,label = self.data[x]
        file = np.load(filename)
        if file.shape[0] >= self.length:
            file = file[:self.length]
        else:
            to_append = np.zeros((self.length - file.shape[0],file.shape[1]))
            file = np.concatenate([file,to_append],axis=0)
        return file, label


def get_vocaldata(root):
    """
    get the Vocaldata dataset.
    :param root: directory to the dataset
    """
    return VocalData(root)
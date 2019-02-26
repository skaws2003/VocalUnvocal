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
    def __init__(self,root):
        """
        :param root: directory to the dataset.
        """
        self.root = root
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
        return file, label

def get_vocaldata(root):
    """
    get the Vocaldata dataset.
    :param root: directory to the dataset
    """
    return VocalData(root)
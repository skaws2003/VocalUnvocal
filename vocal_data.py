"""
a dataset for VocalUnvocal project.
"""

import torch
import torchvision
import os
import numpy as np

class VocalData(torch.utils.data.Dataset):
    """
    A pytorch dataset class for vocal dataset.
    """
    def __init__(self,root,max_length=20000):
        """
        :param root: directory to the dataset.
        :param length: length of mfcc features to look at
        """
        self.root = root
        self.max_length = max_length
        on_vocal = os.listdir(os.path.join(self.root,'on'))
        off_vocal = os.listdir(os.path.join(self.root,'off'))
        
        self.data = []
        for f in on_vocal:
            filedir = os.path.join(self.root,'on',f)
            self.data.append((filedir,0))
        for f in off_vocal:
            filedir = os.path.join(self.root,'off',f)
            self.data.append((filedir,1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        filename,label = self.data[index]
        file = np.load(filename)
        if file.shape[0] >= self.max_length:
            file = file[:self.max_length]
            file_len = self.max_length
        else:
            to_append = np.zeros((self.max_length - file.shape[0],file.shape[1]))
            file_len = file.shape[0]
            file = np.concatenate([file,to_append],axis=0)
        return file, label, file_len


def get_vocaldata(root,length):
    """
    get the Vocaldata dataset.
    :param root: directory to the dataset
    """
    return VocalData(root,length)


def test():
    ds = VocalData(root= '/media/skaws2003/HDD/datasets/VOCALS')
    dl = torch.utils.data.DataLoader(dataset=ds,batch_size=2,shuffle=True,num_workers=0)
    it = iter(dl)
    for i in range(5):
        dat = next(it)
        print(dat[2])


if __name__ == '__main__':
    test()
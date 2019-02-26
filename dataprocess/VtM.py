"""
This code converts wave file to mfcc featurs.
"""


from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import numpy as np

WAV_PATH = '/media/skaws2003/HDD/datasets/VOCAL'
MFCC_PATH = '/media/skaws2003/HDD/datasets/VOCAL_mfcc'

wav_files = os.listdir(WAV_PATH)
num = len(wav_files)

for i,f in enumerate(wav_files):
    rate,sig = wav.read(os.path.join(WAV_PATH,f))
    mfcc_feat = mfcc(sig,rate)
    np.save(os.path.join(MFCC_PATH,f[:-4]),mfcc_feat)
    if i % 20 == 0:
        print("%d/%d"%(i,num))
    

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from eeg import EEG


def load_eeg_dataset(path, srate = 128, verbose=False):
    return [load_eeg(file_path, srate, verbose) for file_path in glob.glob(path)]

def order_asc_by_subject(eggs):
    return [sorted(eegs, key=lambda eeg: (int(eeg.subject),int(eeg.resting_state)))]

def to_data_frame(eegs):
    return pd.DataFrame([eeg.to_dict() for eeg in eegs], columns = eegs[0].to_dict().keys())


def load_eeg(file_path, srate = 128, verbose=False):
    datos    = pd.read_csv(file_path, sep=',', header=None)
    eeg_data = datos.to_numpy()
    srate    = 128
    ch       = eeg_data.shape[0]
    samples  = eeg_data.shape[1]
    filename = Path(file_path).stem
    filename_parts = filename.split('_')
    
    if verbose:
        print("Sampling rate: {:.2f} Hz".format(srate))
        print("Data shape: {:d} samples x {:d} channels".format(eeg_data.shape[1], eeg_data.shape[0]))
        print("Tiempo total : {:.2f} ".format(samples/srate))
    
    return EEG(
        np.transpose(eeg_data), 
        ch, 
        samples, 
        srate, 
        filename_parts[1], 
        filename_parts[3]
    )

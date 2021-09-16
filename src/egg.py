import mne
import pandas as pd
import numpy as np
from pathlib import Path

def channels():
    return [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 
        'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2',
        'FC5', 'FC6','CP1','CP2','CP5','CP6','AFz','Fpz','POz'
    ]


def get_info_eeg_and_montage(channels=channels(), montage_type='standard_1020', sfreq = 128):
    montage = mne.channels.make_standard_montage(montage_type)

    ind = [i for (i, channel) in enumerate(montage.ch_names) if channel in channels]
    montage_30 = montage.copy()
    # Me quedo solo con los canales seleccionados
    montage_30.ch_names = [montage.ch_names[x] for x in ind]
    kept_channel_info    = [montage.dig[x+3] for x in ind]

    # Me quedo con las referencias desde dónde están ubicados los canales
    montage_30.dig = montage.dig[0:3]+kept_channel_info

    info_eeg= mne.create_info(
        ch_names = montage_30.ch_names, 
        sfreq    = sfreq,
        ch_types = 'eeg'
    ).set_montage(montage_30)

    return info_eeg, montage_30
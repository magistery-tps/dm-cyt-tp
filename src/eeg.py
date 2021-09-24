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

    
class EEG:
    def __init__(self, data, nchannels, nsamples, sfrequency, subject, resting_state):
        self.data          = data
        self.nchannels     = nchannels
        self.nsamples      = nsamples
        self.sfrequency    = sfrequency
        self.subject       = subject
        self.resting_state = resting_state
        
    def dataT(self): return np.transpose(self.data)

    def to_dict(self):
        return {
            'subject'      : self.subject,
            'resting_state': self.resting_state,
            'nchannels'    : self.nchannels,
            'sfrequency'   : self.sfrequency,
            'nsamples'     : self.nsamples,
            'data'         : self.data
        }


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

def eegs_total_mean(eegs, inicio = 0, fin= 60):
    promedios_totales     = np.empty((0, eegs[0].nchannels))
    
    for eeg in eegs:
        promedios         = eeg.data[inicio * eeg.sfrequency : fin * eeg.sfrequency, :].mean(axis=0)
        promedios_totales = np.concatenate((promedios_totales, promedios.reshape(1,30)), axis=0) 

    return promedios_totales


def order_asc_by_subject(eegs):
    return sorted(eegs, key=lambda eeg: (int(eeg.subject),int(eeg.resting_state)))

def order_asc_by_resting_state(eegs):
    return sorted(eegs, key=lambda eeg: (int(eeg.resting_state),int(eeg.subject)))
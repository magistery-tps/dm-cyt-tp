import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
import matplotlib.gridspec as gridspec
import mne
from sklearn import decomposition

def plot_df_pca(df, info_eeg, n_components=3, figsize=(10, 3),title_size=15, title_y=1.05):
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(np.transpose(df))
    pcs = pca.transform(np.transpose(df))
    
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(ncols=pcs.shape[1], figsize=figsize)

    for p in range(pcs.shape[1]):
      mne.viz.plot_topomap(
          pcs[:,p],
          info_eeg,
          cmap='coolwarm', 
          contours=0,
          axes=ax[p],
          show=False
      )
    fig.suptitle(
        f'{round(var[p]*100, 2)}% de varianza explicada por las primeras {n_components} componentes.',
        size=title_size,
        y=title_y
    )

def plot_eeg_pca(info_eeg, eeg, n_components=3, figsize=(10, 3),title_size=15, title_y=1.05):
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(eeg.dataT())
    pcs = pca.transform(eeg.dataT())

    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(ncols=pcs.shape[1], figsize=figsize)

    for p in range(pcs.shape[1]):
        mne.viz.plot_topomap(
            pcs[:,p],
            info_eeg,
            cmap='coolwarm', 
            contours=0,
            axes=ax[p],
            show=False
        )
    fig.suptitle(
        f'{round(var[p]*100, 2)}% de varianza explicada por las primeras {n_components} componentes.',
        size=title_size,
        y=title_y
    )
    


def plot_eeg_topology(data_prom, info_eeg, vminimo, vmaximo, figsize=(8, 4)):
    _, axis = plt.subplots(1, figsize=figsize)
    plot_eeg_topology_on_axis(axis, data_prom, info_eeg, vminimo, vmaximo)


def plot_eeg_topology_on_axis(axis, data_prom, info_eeg, vminimo, vmaximo):
    im, cm  = mne.viz.plot_topomap(
        data_prom,
        info_eeg,
        vmin     = vminimo,
        vmax     = vmaximo,
        cmap     = 'coolwarm', 
        contours = 0, 
        show     = True
    )
    axis.set_title('Topogragía Promedio')



def plot_egg(eeg, montage, inicio = 1, fin= 10, figsize=(14, 10)):
    _, axis = plt.subplots(1, figsize=figsize)
    plot_egg_on_axis(axis, eeg, montage, inicio, fin)
    
def plot_egg_on_axis(
    axis,
    eeg,
    montage,
    inicio = 1,
    fin    = 10
):
    start = inicio * eeg.sfrequency
    end   = fin * eeg.sfrequency
    
    xticks = np.arange(
        start,
        (fin + 1) * eeg.sfrequency,
        eeg.sfrequency
    )
    yticks=[]

    
    for channel in np.arange(eeg.nchannels):
        temp = eeg.dataT()[channel,start:end]
        
        dmin = np.min(temp)
        dmax = np.max(temp)

        vmedio = np.mean([dmin, dmax]) + 30 * channel
        
        yticks.append(vmedio)
        
        axis.plot(
            np.arange(start, end), 
            vmedio * np.ones_like(temp) + temp,
            'k'
        )

    axis.set_xlim([start, end])
    axis.set_xticks(xticks)
    axis.set_xticklabels(np.arange(inicio, fin + 1))
    axis.set_yticks(yticks)
    axis.set_yticklabels(montage.ch_names)
    axis.set_ylabel('Canales')
    axis.set_xlabel('Tiempo (Seg.)')
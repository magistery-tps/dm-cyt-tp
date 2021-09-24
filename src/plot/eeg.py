import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
import matplotlib.gridspec as gridspec
import mne
from sklearn import decomposition

def plot_eeg_pca(info_eeg, eeg_data, n_components=3, figsize=(10, 3),title_size=15, title_y=1.05):
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(eeg_data)
    pcs = pca.transform(eeg_data)

    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(ncols=pcs.shape[1], figsize=figsize)
    for p in range(pcs.shape[1]):
        mne.viz.plot_topomap(
            pcs[:,p],
            info_eeg,
            cmap     ='coolwarm', 
            contours = 0,
            axes     = ax[p],
            show     = False
        )
    fig.suptitle(
        f'{round(var[p]*100, 2)}% de varianza explicada por las primeras {n_components} componentes.',
        size=title_size,
        y=title_y
    )


def plot_eeg_topology(data_prom, info_eeg, vminimo, vmaximo, figsize=(8, 4)):
    _, axis = plt.subplots(1, figsize=figsize)
    plot_eeg_topology_on_axis(axis, data_prom, info_eeg, vminimo, vmaximo)


def plot_eeg_topology_on_axis(axis, data_prom, info_eeg, vminimo, vmaximo, title_size=16, title_y=1):
    axis.set_title(
        'Topografía Promedio',
        size=title_size,
        y=title_y
    )
    mne.viz.plot_topomap(
        data_prom,
        info_eeg,
        vmin     = vminimo,
        vmax     = vmaximo,
        cmap     = 'coolwarm', 
        contours = 0, 
        show     = True,
        axes     = axis
    )
    


def plot_egg(eeg, montage, inicio = 1, fin= 10, figsize=(14, 10), title_size=18, title_y=1):
    _, axis = plt.subplots(1, figsize=figsize)
    plot_egg_on_axis(axis, eeg, montage, inicio, fin, title_size, title_y)
    
def plot_egg_on_axis(
    axis,
    eeg,
    montage,
    inicio = 1,
    fin    = 10,
    title_size=16, 
    title_y=1
):
    axis.set_title(
        'Señales por sensor',
        size=title_size,
        y=title_y
    )
        
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


def plot_eegs_summary(eegs, info_eeg, montage, inicio = 0, fin= 60):
    if len(eegs) == 0: return 

    nchannels             = eegs[0].nchannels
    datos_orig            = np.empty((0, nchannels + 2))  #30 canales + 2 metadata individuo y tipo de reposo
    promedios_totales     = np.empty((0, nchannels))
    inicio                = 1 #calcula los promedios entre 1 y 4
    fin                   = 4

    for eeg in sorted(eegs, key=lambda it: it.subject + it.resting_state):
        participante = np.repeat(int(eeg.subject), eeg.nsamples).reshape(eeg.nsamples, 1)
        tipo       = np.repeat(int(eeg.resting_state), eeg.nsamples).reshape(eeg.nsamples, 1)    

        metadata   = np.concatenate((participante,tipo),axis=1)
        datos_orig = np.concatenate((datos_orig,np.concatenate((eeg.data, metadata),axis=1)),axis=0)

        promedios         = eeg.data[inicio*eeg.sfrequency:fin*eeg.sfrequency, :].mean(axis=0)
        promedios_totales = np.concatenate((promedios_totales, promedios.reshape(1,30)), axis=0) 

        fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [3, 1.2]})
        fig.suptitle(f'Sujeto {eeg.subject} - Estado de reposo {eeg.resting_state}', size=20, y=1.05)

        plot_egg_on_axis(axes[0], eeg, montage)
        plot_eeg_topology_on_axis(
            axes[1],
            promedios,
            info_eeg,
            -0.03,  # Estos son los valores máximo y minimo de todos los promedios.
            0.02 
        )
        plot_eeg_pca(info_eeg, eeg.dataT())

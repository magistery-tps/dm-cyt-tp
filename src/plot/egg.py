import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
import matplotlib.gridspec as gridspec
import mne
from sklearn import decomposition

def plot_df_pca(df, info_eeg, n_components=3):
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(np.transpose(df))
    pcs = pca.transform(np.transpose(df))
    
    print('PCS:', pcs.shape)
    
    var = pca.explained_variance_ratio_
    
    fig2, ax = plt.subplots(
        ncols=pcs.shape[1], 
        figsize=(10, 3), 
        gridspec_kw=dict(top=0.9),
        sharex=True, 
        sharey=True
    )

    for p in range(pcs.shape[1]):
      mne.viz.plot_topomap(
          pcs[:,p],
          info_eeg,
          cmap='coolwarm', 
          contours=0,
          axes=ax[p],
          show=False
      )
      ax[p].set_title('var:'+str(round(var[p]*100,2)) )
    
    return var

def plot_eeg_pca(info_eeg, eeg_data, n_components=3):
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(np.transpose(eeg_data))

    pcs = pca.transform(np.transpose(eeg_data))
    print(pcs.shape)

    var = pca.explained_variance_ratio_
    fig2, ax = plt.subplots(
        ncols=pcs.shape[1], 
        figsize=(10, 3), 
        gridspec_kw=dict(top=0.9),
        sharex=True, sharey=True
    )
    
    for p in range(pcs.shape[1]):
        mne.viz.plot_topomap(
            pcs[:,p],
            info_eeg,
            cmap='coolwarm', 
            contours=0,
            axes=ax[p],
            show=False
        )
    ax[p].set_title('var:'+str(round(var[p]*100,2)) )

    
def plot_eeg_topology(
    data_prom,
    info_eeg,
    vminimo,
    vmaximo
):
    fig, ax = plt.subplots(figsize=(8, 4), gridspec_kw=dict(top=0.9), sharex=True, sharey=True)
    im,cm  = mne.viz.plot_topomap(
        data_prom,
        info_eeg,
        vmin=vminimo,
        vmax=vmaximo,
        cmap='coolwarm', 
        contours=0, 
        show=True
    )
    ax.set_title('topogragía promedio')
    ax_x_start = 0.95
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.9
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    #clb.ax.set_title(unit_label,fontsize=fontsize) # title on top of colorbar
    # plt.show()  


def plot_egg(
    eeg_data, 
    montage, 
    srate,
    inicio = 1,
    fin    = 10
):
    ch     = eeg_data.shape[0]    
    xticks = np.arange(inicio*srate,(fin+1)*srate,srate)
    fig,ax = plt.subplots(ncols=1,figsize=(16,8))
    #fig.suptitle('Series temporales (uV)')
    yticks=[]

    for c in np.arange(ch):
        temp = eeg_data[c,inicio*srate:fin*srate]
        dmin=np.min(temp)
        dmax=np.max(temp)
        vmedio = np.mean([dmin,dmax])+30*c
        yticks.append(vmedio)
        ax.plot(np.arange(inicio*srate,fin*srate),vmedio*np.ones_like(temp)+temp,'k')

    ax.set_xlim([inicio*srate,fin*srate])
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.arange(inicio,fin+1))
    ax.set_yticks(yticks)
    ax.set_yticklabels(montage.ch_names)
    ax.set_ylabel('channels')
    ax.set_xlabel('Time (s)')
    
    plt.show()

def plot_egg_diff(
    eeg_data_a, 
    eeg_data_b, 
    srate, 
    montage,
    inicio = 1,
    fin    = 4
):
    ch   = eeg_data.shape[0]

    xticks = np.arange(inicio*srate,(fin+1)*srate,srate)
    fig,ax = plt.subplots(ncols=1,figsize=(16,8))
    #fig.suptitle('Series temporales (uV)')
    yticks=[]

    for c in np.arange(ch):
        temp   = eeg_data_a[c,inicio*srate:fin*srate]
        temp2  = eeg_data_b[c,inicio*srate:fin*srate]
        dmin   = np.min(temp)
        dmax   = np.max(temp)
        vmedio = np.mean([dmin,dmax])+30*c
        yticks.append(vmedio)
        ax.plot(np.arange(inicio*srate,fin*srate),vmedio*np.ones_like(temp)+temp,'--r')
        ax.plot(np.arange(inicio*srate,fin*srate),vmedio*np.ones_like(temp2)+temp2,'k')

    ax.set_xlim([inicio*srate,fin*srate])
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.arange(inicio,fin+1))
    ax.set_yticks(yticks)
    ax.set_yticklabels(montage.ch_names)
    ax.set_ylabel('channels')
    ax.set_xlabel('Time (s)')
    
    plt.show()

    fig,ax=plt.subplots(ncols=1,figsize=(16,2))
    ax.plot(np.arange(inicio*srate,fin*srate),vmedio*np.ones_like(temp)+temp,'--r')
    ax.plot(np.arange(inicio*srate,fin*srate),vmedio*np.ones_like(temp2)+temp2,'k')
    ax.set_xlim([inicio*srate,fin*srate])
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.arange(inicio,fin+1))
    ax.set_yticks([860])
    ax.set_yticklabels(mont1020_30.ch_names[c-1])
    ax.set_ylabel('channels')
    ax.set_xlabel('Time (s)')

    plt.show()
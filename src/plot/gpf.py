import numpy as np
import matplotlib.pylab as plt 
import matplotlib.gridspec as gridspec

def plot_GPF(GFP, peaks, t):
    fig,ax=plt.subplots(ncols=1,figsize=(16,4))
    xticks = np.arange(t[0],t[-1],10)
    ax.plot(t,GFP)
    ax.plot(t[peaks],GFP[peaks],'co')
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.arange(t[0],t[-1],10))
    ax.set_xticks(xticks)
    ax.set_ylabel('GCP')
    ax.set_xlabel('Time (s)')
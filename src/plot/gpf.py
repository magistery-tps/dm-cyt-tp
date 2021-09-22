import numpy as np
import matplotlib.pylab as plt 
import matplotlib.gridspec as gridspec

def plot_gfp(gfp, peaks, sfrequency, figsize=(30, 8)):
    t = np.arange(0, gfp.shape[0]/sfrequency, 1/sfrequency)

    fig, ax = plt.subplots(ncols=1, figsize=figsize)

    xticks = np.arange(t[0], t[-1], 10)

    ax.plot(t, gfp)
    ax.plot(t[peaks], gfp[peaks],'co')
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.arange(t[0], t[-1], 10))
    ax.set_xticks(xticks)
    
    ax.set_ylabel('GCP')
    ax.set_xlabel('Time (s)')

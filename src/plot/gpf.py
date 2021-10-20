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
    
    
def plot_sx_reduce_noise(X,X_new, nro_canal):
    import matplotlib.pylab as plt 
    plt.scatter(np.arange(0,len(X[nro_canal,:])),X[nro_canal,:],alpha=0.8,marker="+")
    plt.scatter(np.arange(0,len(X[nro_canal,:])),X_new[nro_canal, :], alpha=0.2,marker="x")
    plt.ylabel("amplitud")
    plt.xlabel("muestras")
    plt.title("Canal:"+str(nro_canal))
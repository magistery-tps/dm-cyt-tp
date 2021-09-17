import numpy as np
import matplotlib.pylab as plt 
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

def plot_GFP(GFP, peaks, t):
    fig,ax=plt.subplots(ncols=1,figsize=(16,4))
    xticks = np.arange(t[0],t[-1],10)
    ax.plot(t,GFP)
    ax.plot(t[peaks],GFP[peaks],'co')
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.arange(t[0],t[-1],10))
    ax.set_xticks(xticks)
    ax.set_ylabel('GCP')
    ax.set_xlabel('Time (s)')


def extraction_gfp(eeg, prominence,distance,height):
  u_mean = np.mean(eeg.dataT(), axis=0)  
  v_i = eeg.dataT() - u_mean
  gfp=np.sqrt(1/30*np.sum(v_i**2,axis=0)) 
  peaks, _ = find_peaks(gfp, height=height, prominence=prominence,distance=distance)
  v_i_peaks=v_i[:,peaks]
  print("Cantidad de picos de GFP:", len(peaks))
  print("El intervalo temporal entre máximos de GFP promedio es:",np.mean(np.diff(peaks))/eeg.sfrequency,'ms')
  print("El intervalo temporal entre máximos de GFP más chico es:",min(np.diff(peaks))/eeg.sfrequency,'ms')
  print("El intervalo temporal entre máximos de GFP más grande es:",max(np.diff(peaks))/eeg.sfrequency,'ms')
  return gfp, peaks, v_i_peaks

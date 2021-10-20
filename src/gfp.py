from plot import plot_gfp
import numpy as np
from scipy.signal import find_peaks


def vi(eeg_data):
    u_mean = np.mean(eeg_data.T, axis=0)  
    return eeg_data.T - u_mean

def gfp(eeg_data):
    return np.sqrt(1/30 * np.sum(vi(eeg_data) ** 2 , axis=0))

def find_gfp_peaks(eeg_data, prominence, distance, height, outliers, n_components):
    total_gfp = gfp(eeg_data)

    peaks, _ = find_peaks(
        total_gfp,
        height     = height, 
        prominence = prominence,
        distance   = distance
    )
    
    v_i = vi(eeg_data)
    v_i_peaks = v_i[:, peaks]
    gfp_values = total_gfp[peaks]
    gfp2 = np.sum(gfp_values**2) # normalizing constant in GEV
    n_gfp = peaks.shape[0]
    return total_gfp, peaks, v_i_peaks, gfp_values, gfp2, n_gfp 


def show_gfp_peaks_summary(gfp, peaks, sfrequency):
    print("- Cantidad de picos de GFP:", len(peaks))
    print(
        "- El intervalo temporal entre máximos de GFP promedio es:",
        np.mean(np.diff(peaks))/sfrequency*1000,
        'ms'
    )
    print(
        "- El intervalo temporal entre máximos de GFP más chico es:", 
        min(np.diff(peaks)) / sfrequency*1000,
        'ms'
    )
    print(
        "- El intervalo temporal entre máximos de GFP más grande es:",
        max(np.diff(peaks)) / sfrequency*1000,
        'ms'
    )

def show_complete_gfp_peaks_summary(gfp, peaks, sfrequency):
    show_gfp_peaks_summary(gfp, peaks, sfrequency)
    plot_gfp(gfp, peaks, sfrequency)
    
    
def reduce_noise_pca(X,n_comp_pples):
    from sklearn import decomposition
    pca = decomposition.PCA(n_components=n_comp_pples)
    pca.fit(X)
    X_pca = pca.transform(X)
    X_new = pca.inverse_transform(X_pca)
    exp_var_cumul = np.cumsum(pca.fit(X).explained_variance_ratio_)
    return X_new, exp_var_cumul

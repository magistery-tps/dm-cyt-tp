import numpy as np
from gfp import vi

def calculo_gev(X, maps, n_ch, n_clusters, gfp, gfp2, L):
    data_norm = X.T
    #data_norm = data - data.mean(axis=1, keepdims=True)
    #data_norm /= data_norm.std(axis=1, keepdims=True)
    n_t = data_norm.shape[0]
    # --- GEV ---
    maps_norm = maps - maps.mean(axis=1, keepdims=True)
    #maps_norm /= maps_norm.std(axis=1, keepdims=True)
    maps_norm /= np.sqrt(np.sum(maps**2, axis=1, keepdims=True))
    # --- correlation data, maps ---
    C = np.dot(data_norm, maps_norm.T)/n_ch
    print("C.shape: " + str(C.shape[0])+str(C.shape[1]))
    #print("C.min: {C.min():.2f}   Cmax: {C.max():.2f}")

    # --- GEV_k & GEV ---
    gev = np.zeros(n_clusters)
    for k in range(n_clusters):
        r = L==k
        gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2
    
    gev_sum=np.sum(gev)
    #print(f"\n[+] Global explained variance GEV = {gev_sum:.3f}")
    #for k in range(n_clusters):
    #    print(f"GEV_{k:d}: {gev[k]:.3f}")
    
    return gev, gev_sum
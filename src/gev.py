import numpy as np
from gfp import vi

def calculo_gev(eeg_data, maps, n_ch, n_clusters, gfp, gfp2):
  data_norm = vi(eeg_data).T
  #data_norm = data - data.mean(axis=1, keepdims=True)
  data_norm /= data_norm.std(axis=1, keepdims=True)
  n_t = data_norm.shape[0]
  L = np.zeros(n_t)
  # --- GEV ---
  maps_norm = maps - maps.mean(axis=1, keepdims=True)
  maps_norm /= maps_norm.std(axis=1, keepdims=True)
  # --- correlation data, maps ---
  C = np.dot(data_norm, maps_norm.T)/n_ch
  #print("C.shape: " + str(C.shape))
  #print("C.min: {C.min():.2f}   Cmax: {C.max():.2f}")

  # --- GEV_k & GEV ---
  gev = np.zeros(n_clusters)
  for k in range(n_clusters):
      r = L==k
      gev[k] = np.sum(gfp[r]**2 * C[r,k]**2)/gfp2
  print(f"\n[+] Global explained variance GEV = {gev.sum():.3f}")
  for k in range(n_clusters):
      print(f"GEV_{k:d}: {gev[k]:.3f}")
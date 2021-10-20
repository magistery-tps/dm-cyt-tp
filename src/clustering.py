import numpy as np

    
# k means modificado 
def kmeans2(gfp_maps,gfp_eval,gfp2, n_maps, n_runs=10, maxerr=1e-6, maxiter=500):
     # los inputs son los mapas, el valor del gfp en los picos,
      # la cantidad de clusters, la cantidad de corridas,
      # error máximo, número máximo de iteraciones
    V = gfp_maps.T
    n_gfp = V.shape[0]
    n_ch = V.shape[1]
    sumV2 = np.sum(V**2)

    # Guarda resultados de cada corrida
    cv_list =   []  # cross-validation criterion for each k-means run
    maps_list = []  # microstate maps for each k-means run
    L_list =    []  # microstate label sequence for each k-means run
    gev_list =  []
  
    for run in range(n_runs):
      # initialize random cluster centroids 
        rndi = np.random.permutation(n_gfp)[:n_maps]
        maps = V[rndi, :]
        # normalize row-wise (across EEG channels)
        maps /= np.sqrt(np.sum(maps**2, axis=1, keepdims=True))
        # initialize
        n_iter = 0
        var0 = 1.0
        var1 = 0.0
        # convergence criterion: variance estimate (step 6)
        while ( (np.abs((var0-var1)/var0) > maxerr) & (n_iter < maxiter) ):
          # (step 3) microstate sequence (= current cluster assignment)
            C = np.dot(V, maps.T)
            C /= (n_ch*np.outer(gfp_eval, np.std(maps, axis=1)))
            L = np.argmax(C**2, axis=1)
            # (step 4)
            for k in range(n_maps):
                Vt = V[L==k, :]
                # (step 4a)
                Sk = np.dot(Vt.T, Vt)
                # (step 4b)
                evals, evecs = np.linalg.eig(Sk)
                v = evecs[:, np.argmax(np.abs(evals))]
                v = v.real
                maps[k, :] = v/np.sqrt(np.sum(v**2))
                # (step 5)
                var1 = var0
                var0 = sumV2 - np.sum(np.sum(maps[L, :]*V, axis=1)**2)
                var0 /= (n_gfp*(n_ch-1))
                n_iter += 1
                if (n_iter > maxiter):
                     print((f"\tK-means run {run+1:d}/{n_runs:d} did NOT converge "
                   f"after {maxiter:d} iterations."))
      # CROSS-VALIDATION criterion for this run (step 8)
        cv = var0 * (n_ch-1)**2/(n_ch-n_maps-1.)**2
        cv_list = np.append(cv_list,cv)
        maps_list.append(maps)
        L_list.append(L)
        # --- GEV_k & GEV ---
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            r = L==k
            gev[k] = np.sum(gfp_eval[r]**2 * C[r,k]**2)/gfp2
        
        gev.total=np.sum(gev)
        print(f"\n[+] Global explained variance GEV = {np.sum(gev):.3f}")
        for k in range(n_clusters):
            print(f"GEV_{k:d}: {gev[k]:.3f}")
        gev_list.append(gev)
     
    # select best run. Lo elige en función del validación cruzada
    k_opt = np.argmin(cv_list)
    maps = maps_list[k_opt]
    L  = L_list[k_opt]
    cv = cv_list[k_opt] 
    gev=gev_list[k_opt]
    return maps, L, cv, gev


def silhoutte_modificado2(maps,data,labels,ch,n_clusters):
    if data.shape[0]!=ch:
        data=data.T
    elif maps.shape[0]!=ch:
        maps=maps.T
    
    corr_ctodos =np.abs(np.corrcoef(data.T))
    sil = []
    for n,i in enumerate(data.T):
        L = labels[n]
        dist = 1-corr_ctodos[n,:]
        dist_=np.delete(dist,n)
        lab_ = np.delete(labels,n)
        prom_dist=[]
        for k in range(n_clusters):
            prom_dist.append(np.mean(dist_[lab_==k]))
        a=prom_dist[L]
        b=np.min(np.delete(prom_dist,L))
        sil.append((b-a)/np.max([a,b]))
    return sil

def van_dongen(ct):  
    n2=2*(sum(ct.apply(sum,axis=1)))
    sumi = sum(ct.apply(np.max,axis=1))
    sumj = sum(ct.apply(np.max,axis=0))
    maxsumi = np.max(ct.apply(sum,axis=1))
    maxsumj = np.max(ct.apply(sum,axis=0))
    vd = (n2 - sumi - sumj)/(n2 - maxsumi - maxsumj)
    return vd
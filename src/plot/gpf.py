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


    
#%% Grafico silhoutte
def grafica_sil(n_clusters,sil,labels):
    import matplotlib.cm as cm
    # Create a subplot with 1 row and 2 columns
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.5, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(labels) + (n_clusters + 1) * 10])

    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = np.mean(sil)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = np.array(sil)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
   # ax1.set_xticks([-1,-0.5,-0.2,0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()
    

#%% Grafico vectores_kmeans_heatmap
def plot_vectores_kmeans(X, labels_gfp, n_clusters):
    import seaborn as sns
    import numpy as np
    import matplotlib.pylab as plt 
    marcasx=np.array(np.where(np.diff(np.sort(labels_gfp))==1))
    plt.figure()
    plt.axes([0, 0, 1, 1])
    sns.heatmap(X[:,np.argsort(labels_gfp)],vmin=-10,vmax=10, cmap='coolwarm')
    plt.title('Vectores ordenados por cluster kmeans mod n=' + str(n_clusters))
    plt.xticks(marcasx[0,:],labels=list(map(str, np.arange(2,n_clusters+1))))
    plt.show()
    

#%% dibuja mapa pca de los clusters
def plot_maps_pca(maps_kmeans,n_clusters,info_eeg):
    import matplotlib.pylab as plt 
    import mne
    fig3, ax = plt.subplots(ncols=n_clusters, figsize=(10, 4), gridspec_kw=dict(top=0.9),sharex=True, sharey=True)
    for n in range(n_clusters):
        mne.viz.plot_topomap(maps_kmeans[n,:].T,info_eeg, vmin=-0.3,vmax=0.3, cmap='coolwarm', contours=0, axes=ax[n],show=False)
    plt.show()
    
    
def plot_sx_reduce_noise(X,X_new, nro_canal):
    import matplotlib.pylab as plt 
    plt.scatter(np.arange(0,len(X[nro_canal,:])),X[nro_canal,:],alpha=0.8,marker="+")
    plt.scatter(np.arange(0,len(X[nro_canal,:])),X_new[nro_canal, :], alpha=0.2,marker="x")
    plt.ylabel("amplitud")
    plt.xlabel("muestras")
    plt.title("Canal:"+str(nro_canal))
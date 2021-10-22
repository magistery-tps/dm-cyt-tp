def asign_cluster(datos,maps,nch):
    import numpy as np  
    # --- correlations of the data sequence with each cluster ---
    m_y =maps.mean(axis=1, keepdims=True)
    s_y = maps.std(axis=1)
    m_x = datos.mean(axis=1, keepdims=True)
    s_x = datos.std(axis=1)
    s_xy = 1.*nch*np.outer(s_x, s_y)
    C = np.dot(datos-m_x, np.transpose(maps-m_y)) / s_xy #da lo mismo que la correlacion comun con corrcoef
    # --- microstate sequence, ignore polarity ---
    L = np.argmax(C**2, axis=1)
    values, counts = np.unique(L, return_counts=True)
    return L, values, counts


def classify_sequence(maps, eegs_subject, n_media):
    import numpy as np
    labels_secuencia_list=list()
    datos_filtrados_list=list()
    sujetos_estados_list=list()

    for i in range(0,10):
        datos=eegs_subject[i].data
        datos_filtrados=np.apply_along_axis(lambda m: np.convolve(m, np.ones(n_media), mode='valid'), axis=0, arr=datos)
        total=datos_filtrados.shape[0]
        labels_secuencia, values, counts=asign_cluster(datos_filtrados,maps_kmeans,30)
        labels_secuencia_list.append(labels_secuencia)
        datos_filtrados_list.append(datos_filtrados)
        sujetos_estados_list.append(np.array([eegs_subject[i].subject,eegs_subject[i].resting_state]))
    return  labels_secuencia_list, datos_filtrados_list, sujetos_estados_list


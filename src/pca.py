import pandas as pd

def to_df(pca_result, labels):
    components = pd.DataFrame(
        data    = pca_result,
        columns = ['pc1', 'pc2']
    )
    labels = pd.DataFrame(
        data    = labels,
        columns = ['label']
    )
    return pd.concat([components, labels], axis = 1)
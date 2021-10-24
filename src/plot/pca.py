import matplotlib.pylab as plt
import numpy as np

def plot_pca_df(df, title = '', target_column = 'label'):
    targets = np.sort(df.label.unique())
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC 1', fontsize = 15)
    ax.set_ylabel('PC 2', fontsize = 15)
    ax.set_title(title, fontsize = 12)

    for target in targets:
        indicesToKeep = df[target_column] == target
        ax.scatter(
            df.loc[indicesToKeep, 'pc1'],
            df.loc[indicesToKeep, 'pc2'],
            label=target,
            s = 10
        )
    ax.legend(targets, title='Clusters')
    ax.grid()

import matplotlib.pylab as plt
import numpy as np

def plot_pca_df(df, title = '', target_column = 'label'):
    targets = np.sort(df.label.unique())
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('pc1', fontsize = 15)
    ax.set_ylabel('pc2', fontsize = 15)
    ax.set_title(title, fontsize = 20)
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets, colors):
        indicesToKeep = df[target_column] == target
        ax.scatter(
            df.loc[indicesToKeep, 'pc1'],
            df.loc[indicesToKeep, 'pc2'],
            c = color,
            s = 50
        )
    ax.legend(targets)
    ax.grid()

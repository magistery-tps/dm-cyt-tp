import matplotlib.pylab as plt

def plot_pca_df(df, title = '', target_column = 'label', targets = []):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC 1', fontsize = 15)
    ax.set_ylabel('PC 22', fontsize = 15)
    ax.set_title(title, fontsize = 20)
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = df[target_column] == target
        ax.scatter(
            df.loc[indicesToKeep, 'pc 1'],
            df.loc[indicesToKeep, 'pc 2'],
            c = color,
            s = 50
        )
    ax.legend(targets)
    ax.grid()

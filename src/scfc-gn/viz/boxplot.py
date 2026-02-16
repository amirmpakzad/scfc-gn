def boxplot(data, tick_labels, title, ylabel="Spearman rho"):

    fig, ax = plt.subplots()

    bp = ax.boxplot(
        data,
        tick_labels=tick_labels,
        showmeans=True,
        patch_artist=True,   
        meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'),
        medianprops=dict(color='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )

    colors = ["#4C72B0", "#DD8452"] 
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
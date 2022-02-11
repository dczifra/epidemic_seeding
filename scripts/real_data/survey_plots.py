import numpy as np
import matplotlib.pyplot as plt

distance_dep = {
    "Közeli ismerős":"K91_01",
    "Lakóhely":"K91_02",
    "Megye":"K91_03",
    "Főváros":"K91_05",
    "Ország":"K91_06",
    "Európa":"K91_08",
    "Világ":"K91_09",
}

size_dep = {
    "Közeli ismerős":"K91_01",
    "Lakóhely":"K91_02",
    "Közeli nagyváros":"K91_04",
    "Főváros":"K91_05",
}

def plot_distance_dep(df, criterions, labels, title):
    N = len(distance_dep)
    plt.figure(figsize=(10,7))
    
    for label,criterion in zip(labels, criterions):
        dist_group = [df[criterion].groupby([key])['súly'].sum() for key in distance_dep.values()]
        plt.plot(range(N), [np.average(x.index, weights=x.array) for x in dist_group], label=label)
        #mean=[np.average(x.index, weights=x.array) for x in dist_group]
        #err=[np.sqrt(np.cov(x.index, aweights=x.array)) for x in dist_group]
        #plt.errorbar(range(N), mean, yerr = err, label=label, fmt='-o')
        
    plt.xticks(range(N), labels=distance_dep.keys(), rotation=45, ha='right');
    plt.legend()
    plt.ylabel("Awareness")
    plt.title(title)
    plt.show()
    
def plot_distance_dep2(df, criterions1, criterions2, labels, title):
    N = len(distance_dep)
    plt.figure(figsize=(15,7))
    
    # Country
    plt.subplot(1,2,1)
    for label,criterion in zip(labels, criterions1):
        dist_group = [df[criterion].groupby([key])['súly'].sum() for key in distance_dep.values()]
        plt.plot(range(N), [np.average(x.index, weights=x.array) for x in dist_group], label=label)
        
    plt.xticks(range(N), labels=distance_dep.keys(), rotation=45, ha='right');
    plt.legend()
    plt.title(title+" (noBP)")
    
    # Budapest
    distance_dep_bp = dict(distance_dep)
    del distance_dep_bp["Megye"]
    
    plt.subplot(1,2,2)
    for label,criterion in zip(labels, criterions2):
        dist_group = [df[criterion].groupby([key])['súly'].sum() for key in distance_dep_bp.values()]
        plt.plot(range(N-1), [np.average(x.index, weights=x.array) for x in dist_group], label=label)
        
    plt.xticks(range(N-1), labels=distance_dep_bp.keys(), rotation=45, ha='right');
    plt.legend()
    plt.title(title+" (BP)")
    
    plt.show()
    
def plot_size_dep(df, criterions, labels, title):
    N = len(size_dep)
    plt.figure(figsize=(8,7))
    
    for label,criterion in zip(labels, criterions):
        dist_group = [df[criterion].groupby([key])['súly'].sum() for key in size_dep.values()]
        plt.plot(range(N), [np.average(x.index, weights=x.array) for x in dist_group], label=label)
        
    plt.xticks(range(N), labels=size_dep.keys(), rotation=45, ha='right');
    plt.legend()
    plt.title(title)
    plt.show()

def survey(results, category_names, figsize=(12,8)):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    #category_colors = plt.get_cmap('RdYlGn')(
    category_colors = plt.get_cmap('seismic')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.1 else 'black'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 0.95),
              loc='lower left', fontsize='small')

    return fig, ax

def mysurvey(df, Q_key, Q_name, category_names, figsize=(12,8)):
    results = {}
    for key,name in zip(Q_key, Q_name):
        x = df.groupby([key])['súly'].sum()
        results[name]=np.round(100*x.array/np.sum(x.array), decimals=2)
    survey(results, category_names, figsize)
    
def critsurvey(df, Q_key, Q_name, category_names, criterions, figsize=(12,8)):
    results = {}
    for key,name,crit in zip(Q_key, Q_name, criterions):
        x = df[crit].groupby([key])['súly'].sum()
        results[name]=np.round(100*x.array/np.sum(x.array), decimals=2)
    return survey(results, category_names, figsize)
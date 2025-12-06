import torch as t
import matplotlib.pyplot as plt
import numpy as np
from method import runTrial, softtopk


def plot_good_bad_trajectories(x: t.Tensor, n: int, 
                               num_samples: int = 8,
                               percentile_low: float = 0.25,
                               percentile_high: float = 0.75,
                               fpath = None, title=None):
    """
    x: ExN tensor of trajectories
    n: first n columns are 'bad', remaining are 'good'
    num_samples: number of random example trajectories per group
    percentile_low/high: envelope bounds
    """
    
    E, N = x.shape

    bad = x[:, :n]
    good = x[:, n:]

    print(E,N)
    # ---- Helper to compute summary statistics ----
    def summarize(group):
        mean = group.mean(dim=1)
        lo, hi = t.quantile(group, t.tensor([percentile_low, percentile_high]), dim=1)
        return mean, lo, hi

    mean_g, lo_g, hi_g = summarize(good)
    mean_b, lo_b, hi_b = summarize(bad)


    times = np.arange(E)

    # ---- Plot good ----
    plt.plot(times, mean_g, label="Behaving", linewidth=2)
    plt.fill_between(times, lo_g, hi_g, alpha=0.2, color="blue")

    # ---- Plot bad ----
    plt.plot(times, mean_b, label="Scheming", linewidth=2)
    plt.fill_between(times, lo_b, hi_b, alpha=0.2, color="red")

    # random good samples
    idx = t.randperm(good.shape[1])[:num_samples]
    for i in idx:
        plt.plot(times, good[:, i], alpha=0.2, color="blue")
    # random bad samples
    idx = t.randperm(bad.shape[1])[:num_samples]
    for i in idx:
        plt.plot(times, bad[:, i], alpha=0.2, color="red")

    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    if fpath is None:
        plt.show()
    else:
        plt.savefig(fpath)
        plt.close()

def plot_value_sets(value_sets, legend, colors=None, save_path=None, title=None):
    """
    value_sets: list of lists.
        Each element is a list of item-value sequences.
        Every item-value sequence must be a list or 1D array of length N.
    
    save_path: if not None, saves to PNG instead of showing.
    """
    print(value_sets)
    num_sets = len(value_sets)
    if colors is None:
      colors = plt.cm.tab10(np.linspace(0, 1, num_sets))

    plt.figure(figsize=(8, 5))

    NN = 1/(len(value_sets)+2)
    for set_idx, items in enumerate(value_sets):
        color = colors[set_idx]

        N = len(items)
        x = [x+set_idx*NN for x in np.arange(1, N + 1)]
        #plt.scatter(x, items, color=color, alpha=0.4, label=legend[set_idx], s=60)
        plt.bar(x, items, color=color, alpha=0.4, label=legend[set_idx], width=NN)

        # Optional: darker mean line for each group
        # stacked = np.stack(items, axis=0)
        # mean_line = stacked.mean(axis=0)
        # plt.plot(np.arange(1, N+1), mean_line, color=color, linewidth=3)

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


#plot_value_sets([[1,2,3],[1.1,4,3]])

# estimate, truth, stats = runTrial(40,verbose=False, scale=0.25, iters=40, alpha=0.2, endDiscrim=10)
# print(estimate)

# plot_good_bad_trajectories(stats["weights"].clip(0,1),stats["schemers"])
#print(softtopk(t.tensor([-10,1,2,10,1000],dtype=t.float),3))
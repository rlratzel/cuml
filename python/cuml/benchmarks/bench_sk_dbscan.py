import os

import pandas as pd
import cudf as gd

from sklearn.datasets import make_blobs

from sklearn.metrics import adjusted_rand_score

from sklearn.cluster import DBSCAN as skDBSCAN


#### cell ##
#%%time
def runSkDbscan():
    clustering_sk = skDBSCAN(eps=eps,
                             min_samples=min_samples,
                             algorithm="brute",
                             n_jobs=-1)
    clustering_sk.fit(host_data)
    return clustering_sk

#### cell ##
#%%time
def runComputeScore(clustering_sk):
    return adjusted_rand_score(host_labels, clustering_sk.labels_)


def run():
    #### cell ##
    n_samples = 100000
    n_features = 128

    eps = 3
    min_samples = 2

    #### cell ##
    host_data, host_labels = make_blobs(
           n_samples=n_samples, n_features=n_features, centers=5, random_state=7)

    host_data = pd.DataFrame(host_data)
    host_labels = pd.Series(host_labels)

    #### cell ##
    device_data = gd.DataFrame.from_pandas(host_data)
    device_labels = gd.Series(host_labels)

    #### cell ##
    return runComputeScore(runSkDbscan())


if __name__ == "__main__":
    import sys

    sys.exit(run())

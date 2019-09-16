import os

import pandas as pd
import cudf as gd

from sklearn.datasets import make_blobs

from sklearn.metrics import adjusted_rand_score

from sklearn.cluster import DBSCAN as skDBSCAN
from cuml.cluster import DBSCAN as cumlDBSCAN


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
def runCumlDbscan():
    clustering_cuml = cumlDBSCAN(eps=eps,
                                 min_samples=min_samples)
    clustering_cuml.fit(device_data)
    return clustering_cuml

#### cell ##
#%%time
def runComputeScores(clustering_cuml, clustering_sk):
    cuml_score = adjusted_rand_score(host_labels, clustering_cuml.labels_)
    sk_score = adjusted_rand_score(host_labels, clustering_sk.labels_)
    return (cuml_score, sk_score)


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
    (cuml_score, sk_score) = runComputeScores(runCumlDbscan(), runSkDbscan())
    passed = (cuml_score - sk_score) < 1e-10
    print('compare kmeans: cuml vs sklearn labels_ are ' + ('equal' if passed else 'NOT equal'))


if __name__ == "__main__":
    import sys

    sys.exit(run())

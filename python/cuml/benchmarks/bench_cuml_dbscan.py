import os

import pandas as pd
import cudf as gd

from sklearn.datasets import make_blobs

from sklearn.metrics import adjusted_rand_score

from cuml.cluster import DBSCAN as cumlDBSCAN


#### cell ##
#%%time
def runCumlDbscan():

#### cell ##
#%%time
def runComputeScore(clustering_cuml):


def setup(envObj):
    #### cell ##
    envObj.n_samples = 100000
    envObj.n_features = 128

    envObj.eps = 3
    envObj.min_samples = 2

    #### cell ##
    envObj.host_data, envObj.host_labels = make_blobs(
        n_samples=envObj.n_samples, n_features=envObj.n_features,
        centers=5, random_state=7)

    envObj.host_data = pd.DataFrame(envObj.host_data)
    envObj.host_labels = pd.Series(envObj.host_labels)

    #### cell ##
    envObj.device_data = gd.DataFrame.from_pandas(envObj.host_data)
    envObj.device_labels = gd.Series(envObj.host_labels)


def run(envObj):
    clustering_cuml = cumlDBSCAN(eps=envObj.eps,
                                 min_samples=envObj.min_samples)
    clustering_cuml.fit(envObj.device_data)
    return adjusted_rand_score(envObj.host_labels, clustering_cuml.labels_)


if __name__ == "__main__":
    import sys

    class EnvObj: pass

    envObj = EnvObj()
    setup(envObj)
    sys.exit(run(envObj))

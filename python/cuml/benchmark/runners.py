#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Wrappers to run ML benchmarks"""

from cuml.benchmark import datagen
import time
import numpy as np
import pandas as pd


class SpeedupComparisonRunner:
    """Wrapper to run an algorithm with multiple dataset sizes
    and compute speedup of cuml relative to sklearn baseline."""

    def __init__(
        self, bench_rows, bench_dims, dataset_name='blobs', input_type='numpy'
    ):
        self.bench_rows = bench_rows
        self.bench_dims = bench_dims
        self.dataset_name = dataset_name
        self.input_type = input_type

    def _run_one_size(
        self,
        algo_pair,
        n_samples,
        n_features,
        param_overrides={},
        cuml_param_overrides={},
        cpu_param_overrides={},
        run_cpu=True,
    ):
        data = datagen.gen_data(
            self.dataset_name, self.input_type, n_samples, n_features
        )
        print("data type: ", data[0].__class__)

        cu_start = time.time()
        algo_pair.run_cuml(data, **param_overrides, **cuml_param_overrides)
        cu_elapsed = time.time() - cu_start

        if run_cpu and algo_pair.cpu_class is not None:
            cpu_start = time.time()
            algo_pair.run_cpu(data, **param_overrides)
            cpu_elapsed = time.time() - cpu_start
        else:
            cpu_elapsed = 0.0

        return dict(
            cu_time=cu_elapsed,
            cpu_time=cpu_elapsed,
            speedup=cpu_elapsed / float(cu_elapsed),
            n_samples=n_samples,
            n_features=n_features,
            **param_overrides,
            **cuml_param_overrides
        )

    def run(
        self,
        algo_pair,
        param_overrides={},
        cuml_param_overrides={},
        cpu_param_overrides={},
        *,
        run_cpu=True,
        raise_on_error=False
    ):
        all_results = []
        for ns in self.bench_rows:
            for nf in self.bench_dims:
                try:
                    all_results.append(
                        self._run_one_size(
                            algo_pair,
                            ns,
                            nf,
                            param_overrides,
                            cuml_param_overrides,
                            cpu_param_overrides,
                            run_cpu,
                        )
                    )
                except Exception as e:
                    print(
                        "Failed to run with %d samples, %d features: %s"
                        % (ns, nf, str(e))
                    )
                    if raise_on_error:
                        raise
                    all_results.append(dict(n_samples=ns, n_features=nf))
        return all_results


class AccuracyComparisonRunner(SpeedupComparisonRunner):
    """Wrapper to run an algorithm with multiple dataset sizes
    and compute accuracy and speedup of cuml relative to sklearn
    baseline."""

    def __init__(
        self,
        bench_rows,
        bench_dims,
        dataset_name='blobs',
        input_type='numpy',
        test_fraction=0.10,
    ):
        super().__init__(bench_rows, bench_dims, dataset_name, input_type)
        self.test_fraction = test_fraction

    def _run_one_size(
        self,
        algo_pair,
        n_samples,
        n_features,
        param_overrides={},
        cuml_param_overrides={},
        cpu_param_overrides={},
        run_cpu=True,
    ):
        data = datagen.gen_data(
            self.dataset_name,
            self.input_type,
            n_samples,
            n_features,
            test_fraction=self.test_fraction,
        )
        X_test, y_test = data[2:]

        # rlr
        gpuPollObj = startGpuMetricPolling()
        cu_start = time.time()
        cuml_model = algo_pair.run_cuml(
            data, **{**param_overrides, **cuml_param_overrides}
        )
        cu_elapsed = time.time() - cu_start
<<<<<<< HEAD
        stopGpuMetricPolling(gpuPollObj)

=======
>>>>>>> upstream/branch-0.10
        if algo_pair.accuracy_function:
            if hasattr(cuml_model, 'predict'):
                y_pred_cuml = cuml_model.predict(X_test)
            else:
                y_pred_cuml = cuml_model.transform(X_test)
            cuml_accuracy = algo_pair.accuracy_function(
                y_test, np.asarray(y_pred_cuml)
            )
        else:
            cuml_accuracy = 0.0

        cpu_accuracy = 0.0
        if run_cpu and algo_pair.cpu_class is not None:
            cpu_start = time.time()
            cpu_model = algo_pair.run_cpu(data, **param_overrides)
            cpu_elapsed = time.time() - cpu_start

            if algo_pair.accuracy_function:
                if hasattr(cpu_model, 'predict'):
                    y_pred_cpu = cpu_model.predict(X_test)
                else:
                    y_pred_cpu = cpu_model.transform(X_test)
                cpu_accuracy = algo_pair.accuracy_function(
                    y_test, np.asarray(y_pred_cpu)
                )
        else:
            cpu_elapsed = 0.0

        return dict(
            maxGpuMemUsed=gpuPollObj.maxGpuMemUsed,
            maxGpuUtil=gpuPollObj.maxGpuUtil,
            cu_time=cu_elapsed,
            cpu_time=cpu_elapsed,
            cuml_acc=cuml_accuracy,
            cpu_acc=cpu_accuracy,
            speedup=cpu_elapsed / float(cu_elapsed),
            n_samples=n_samples,
            n_features=n_features,
            **param_overrides,
            **cuml_param_overrides
        )


def run_variations(
    algos,
    dataset_name,
    bench_rows,
    bench_dims,
    param_override_list=[{}],
    cuml_param_override_list=[{}],
    input_type="numpy",
    run_cpu=True,
    raise_on_error=False,
):
    """
    Runs each algo in `algos` once per
    `bench_rows X bench_dims X params_override_list X cuml_param_override_list`
    combination and returns a dataframe containing timing and accuracy data.

    Parameters
    ----------
    algos : str or list
      Name of algorithms to run and evaluate
    dataset_name : str
      Name of dataset to use
    bench_rows : list of int
      Dataset row counts to test
    bench_dims : list of int
      Dataset column counts to test
    param_override_list : list of dict
      Dicts containing parameters to pass to __init__.
      Each dict specifies parameters to override in one run of the algorithm.
    cuml_param_override_list : list of dict
      Dicts containing parameters to pass to __init__ of the cuml algo only.
    run_cpu : boolean
      If True, run the cpu-based algorithm for comparison
    """
    print("Running: \n", "\n ".join([str(a.name) for a in algos]))
    runner = AccuracyComparisonRunner(
        bench_rows, bench_dims, dataset_name, input_type
    )
    all_results = []
    for algo in algos:
        print("Running %s..." % (algo.name))
        for param_overrides in param_override_list:
            for cuml_param_overrides in cuml_param_override_list:
                results = runner.run(
                    algo,
                    param_overrides,
                    cuml_param_overrides,
                    run_cpu=run_cpu,
                    raise_on_error=raise_on_error,
                )
                for r in results:
                    all_results.append(
                        {'algo': algo.name, 'input': input_type, **r}
                    )

    print("Finished all benchmark runs")
    results_df = pd.DataFrame.from_records(all_results)
    print(results_df)

    return results_df



# RLR
import time
import threading
from pynvml import smi

class GPUMetricPoller(threading.Thread):
    def __init__(self, *args, **kwargs):
        self.__stop = False
        super().__init__(*args, **kwargs)
        self.maxGpuUtil = 0
        self.maxGpuMemUsed = 0

    def run(self):
        smi.nvmlInit()
        devObj = smi.nvmlDeviceGetHandleByIndex(0)  # hack - get actual device ID somehow
        memObj = smi.nvmlDeviceGetMemoryInfo(devObj)
        utilObj = smi.nvmlDeviceGetUtilizationRates(devObj)
        initialMemUsed = memObj.used
        initialGpuUtil = utilObj.gpu

        while not self.__stop:
            time.sleep(0.01)

            memObj = smi.nvmlDeviceGetMemoryInfo(devObj)
            utilObj = smi.nvmlDeviceGetUtilizationRates(devObj)

            memUsed = memObj.used - initialMemUsed
            gpuUtil = utilObj.gpu - initialGpuUtil
            if memUsed > self.maxGpuMemUsed:
                self.maxGpuMemUsed = memUsed
            if gpuUtil > self.maxGpuUtil:
                self.maxGpuUtil = gpuUtil

        smi.nvmlShutdown()

    def stop(self):
        self.__stop = True


def startGpuMetricPolling():
    gpuPollObj = GPUMetricPoller()
    gpuPollObj.start()
    return gpuPollObj

def stopGpuMetricPolling(gpuPollObj):
    gpuPollObj.stop()
    gpuPollObj.join()  # consider using timeout and reporting errors

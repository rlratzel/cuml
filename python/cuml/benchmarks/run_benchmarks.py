import argparse
import os
from os import path
import sys

import cuml


from benchmark import (
    Benchmark, logExeTime, printLastResult, noStdoutWrapper, nop
)
from asv_report import updateAsv


########################################
# Update this function to add new algos
########################################
def getBenchmarks():
    benches = [
        Benchmark(name="pagerank",
                  func=cugraph.pagerank,
                  args=(G, args.damping_factor, None, args.max_iter,
                        args.tolerance)),
        Benchmark(name="bfs",
                  func=cugraph.bfs,
                  args=(G, args.source, True)),
        Benchmark(name="sssp",
                  func=cugraph.sssp,
                  args=(G, args.source)),
                  #extraRunWrappers=[noStdoutWrapper]),
        Benchmark(name="jaccard",
                  func=cugraph.jaccard,
                  args=(G,)),
        Benchmark(name="louvain",
                  func=cugraph.louvain,
                  args=(G,)),
        Benchmark(name="weakly_connected_components",
                  func=cugraph.weakly_connected_components,
                  args=(G,)),
        Benchmark(name="overlap",
                  func=cugraph.overlap,
                  args=(G,)),
        Benchmark(name="triangles",
                  func=cugraph.triangles,
                  args=(G,)),
        Benchmark(name="spectralBalancedCutClustering",
                  func=cugraph.spectralBalancedCutClustering,
                  args=(G, 2)),
        Benchmark(name="spectralModularityMaximizationClustering",
                  func=cugraph.spectralModularityMaximizationClustering,
                  args=(G, 2)),
        Benchmark(name="renumber",
                  func=cugraph.renumber,
                  args=(edgelist_gdf["src"], edgelist_gdf["dst"])),
        Benchmark(name="view_adj_list",
                  func=G.view_adj_list),
        Benchmark(name="degree",
                  func=G.degree),
        Benchmark(name="degrees",
                  func=G.degrees),
    ]
    # Return a dictionary of Benchmark name to Benchmark obj mappings
    return dict([(b.name, b) for b in benches])


########################################
# cuml benchmarking utilities


def parseCLI(argv):
    parser = argparse.ArgumentParser(description='CuGraph benchmark script.')
    parser.add_argument('file', type=str,
                        help='Path to the input file')
    parser.add_argument('--algo', type=str, action="append",
                        help='Algorithm to run, must be one of %s, or "ALL"'
                        % ", ".join(['"%s"' % k
                                     for k in getAllPossibleAlgos()]))
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iteration for any iterative '
                        'algo. Default is 100')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Tolerance for any approximation algo. Default '
                        'is 1e-5')
    parser.add_argument('--update_results_dir', type=str,
                        help='Add (and compare) results to the dir specified')
    parser.add_argument('--update_asv_dir', type=str,
                        help='Add results to the specified ASV dir in ASV '
                        'format')
    parser.add_argument('--report_cuda_ver', type=str, default="",
                        help='The CUDA version to include in reports')
    parser.add_argument('--report_python_ver', type=str, default="",
                        help='The Python version to include in reports')
    parser.add_argument('--report_os_type', type=str, default="",
                        help='The OS type to include in reports')
    parser.add_argument('--report_machine_name', type=str, default="",
                        help='The machine name to include in reports')

    return parser.parse_args(argv)


def getAllPossibleAlgos():
    return list(getBenchmarks(nop, nop, nop).keys())

###############################################################################
if __name__ == "__main__":
    perfData = []
    args = parseCLI(sys.argv[1:])

    # set algosToRun based on the command line args
    allPossibleAlgos = getAllPossibleAlgos()
    if args.algo and ("ALL" not in args.algo):
        allowedAlgoNames = allPossibleAlgos + ["ALL"]
        if (set(args.algo) - set(allowedAlgoNames)) != set():
            raise ValueError(
                "bad algo(s): '%s', must be in set of %s" %
                (args.algo, ", ".join(['"%s"' % a for a in allowedAlgoNames])))
        algosToRun = args.algo
    else:
        algosToRun = allPossibleAlgos

    # Update the various wrappers with a list to log to and formatting settings
    # (name width must be >= 15)
    logExeTime.perfData = perfData
    printLastResult.perfData = perfData
    printLastResult.nameCellWidth = max(*[len(a) for a in algosToRun], 15)
    printLastResult.valueCellWidth = 30

    # Load the data file and create a Graph, treat these as benchmarks too
    csvDelim = {"space": ' ', "tab": '\t'}[args.delimiter]
    edgelist_gdf = Benchmark(loadDataFile, args=(args.file, csvDelim)).run()
    G = Benchmark(createGraph, args=(edgelist_gdf, args.auto_csr)).run()

    if G is None:
        raise RuntimeError("could not create graph!")

    print("-" * (printLastResult.nameCellWidth +
                 printLastResult.valueCellWidth))

    benches = getBenchmarks(G, edgelist_gdf, args)

    for algo in algosToRun:
        benches[algo].run()

    #### reports ########################
    if args.update_results_dir:
        raise NotImplementedError

    if args.update_asv_dir:
        # Convert Exception strings in results to None for ASV
        asvPerfData = [(name, value if not isinstance(value, str) else None) for (name, value) in perfData]
        updateAsv(asvDir=args.update_asv_dir,
                  datasetName=args.file,
                  algoRunResults=asvPerfData,
                  cudaVer=args.report_cuda_ver,
                  pythonVer=args.report_python_ver,
                  osType=args.report_os_type,
                  machineName=args.report_machine_name)

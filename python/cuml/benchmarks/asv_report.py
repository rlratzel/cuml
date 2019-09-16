import subprocess
import platform
import os
from os import path
import time
import json

import psutil


def updateAsv(asvDir, datasetName, algoRunResults):
    """
    algoRunResults is a list of (algoName, exeTime) tuples
    """
    if not path.isdir(asvDir):
        os.makedirs(asvDir, exist_ok=True)

    writeMachineInfo(asvDir)

    # Create or update benchmarks.json with the (potentially) new datasetName
    # and algo name(s)
    addBenchmarkInfo(asvDir, datasetName, algoRunResults)

    # Create or update <machine>-<commitHash>.json with the (potentially) new
    # datasetName, algo name(s), and exeTime(s) (results)
    addBenchmarkResults(asvDir, datasetName, algoRunResults)


def writeMachineInfo(asvDir, machineName=""):
    """
    write machine.json to asvDir
    """
    u = platform.uname()
    # FIXME: Error checking
    json.dump({"arch": u.machine,
               "cpu": u.processor,
               "machine": machineName or u.node,
               "os": "%s %s" % (u.system, u.release),
               "ram": "%d" % psutil.virtual_memory().total,
               "version": 1,
               },
              open(path.join(asvDir, "machine.json"), "w"),
              indent=2)


def addBenchmarkInfo(asvDir, datasetName, algoRunResults):
    asvBenchmarkInfo = ASVBenchmarkInfo(asvDir)
    for (algoName, exeTime) in algoRunResults:
        asvBenchmarkInfo.updateParam(benchName=algoName,
                                     paramName="dataset",
                                     paramValue=repr(datasetName))
    asvBenchmarkInfo.write()


def addBenchmarkResults(asvDir, datasetName, algoRunResults, machineName=""):
    commitHash = getCommitHash()
    asvBenchmarkResults = ASVBenchmarkResults(asvDir,
                                              machineName or platform.node(),
                                              commitHash)
    for (algoName, exeTime) in algoRunResults:
        asvBenchmarkResults.updateResults(benchName=algoName,
                                          paramName="dataset",
                                          paramValue=repr(datasetName),
                                          result=exeTime)
    asvBenchmarkResults.write()


def getCommitHash():
    command = "git rev-parse HEAD"
    result = subprocess.run(command.split(), capture_output=True)
    stdout = result.stdout.decode().strip()
    if result.returncode == 0:
        return stdout

    stderr = result.stderr.decode().strip()
    raise RuntimeError("Problem running '%s' (%s - %s)"
                       % (command, stdout, stderr))

def getCudaVer():
    return "10.0"

def getGPUModel():
    return "some GPU"

class ASVBenchmarkInfo:
    """
    """
    fileName = "benchmarks.json"

    def __init__(self, asvDir):
        self.jsonFilePath = path.join(asvDir, self.fileName)
        if path.exists(self.jsonFilePath):
            # FIXME: error checking
            self.__dict = json.load(open(self.jsonFilePath))
        else:
            self.__dict = {}


    def updateParam(self, benchName, paramName, paramValue):
        """
        Update the param lists (param_names and params) for the specified
        benchmark. Creates the corresponding new entries if benchName,
        paramName, or paramValue DNE.
        """
        benchDict = self.__dict.setdefault(benchName,
                                           self.__getDefaultBenchDict(benchName))
        # find paramName in the param_names list to determine which sublist in
        # the params list-of-lists to update.
        param_names = benchDict["param_names"]
        if paramName in param_names:
            paramsIndex = param_names.index(paramName)
            paramsSublist = benchDict["params"][paramsIndex]
            if paramValue not in paramsSublist:
                paramsSublist.append(paramValue)

        # This is a new param so add it to the param_names list and add a new
        # sublist to params containing the new paramValue
        else:
            param_names.append(paramName)
            paramsSublist = benchDict["params"].append([paramValue])


    def write(self):
        json.dump(self.__dict, open(self.jsonFilePath, "w"), indent=2)


    def __getDefaultBenchDict(self, benchName):
        return {"code": "",
                "name": benchName,
                "param_names": [],
                "params": [],
                "timeout": 0,
                "type": "time",
                "unit": "seconds",
                "version": 1,
                }


class ASVBenchmarkResults:
    """
    """
    def __init__(self, asvDir, machineName, commitHash):
        self.jsonFilePath = path.join(asvDir,
                                      "%s-%s.json" % (machineName, commitHash))
        self.machineName = machineName

        if path.exists(self.jsonFilePath):
            # FIXME: error checking
            self.__dict = json.load(open(self.jsonFilePath))
        else:
            self.__dict = {"params": self.__getDefaultTopLevelParams(),
                           "requirements": {},
                           "results": {},
                           "commit_hash": commitHash,
                           "date": time.time(),
                           "python": platform.python_version(),
                           "version": 1,
                           }


    def updateResults(self, benchName, paramName, paramValue, result):
        """
        """
        benchDict = self.__dict["results"].setdefault(
            benchName, self.__getDefaultBenchDict(benchName))

        # FIXME: this is not great
        if not benchDict["params"]:
            benchDict["params"].append([])

        firstList = benchDict["params"][0]
        if paramValue in firstList:
            index = firstList.index(paramValue)
            benchDict["result"][index] = result
        else:
            firstList.append(paramValue)
            index = len(firstList)-1
            benchDict["result"].append(result)


    def write(self):
        json.dump(self.__dict, open(self.jsonFilePath, "w"), indent=2)


    def __getDefaultTopLevelParams(self):
        u = platform.uname()
        return {"cuda": getCudaVer(),
                "gpu": getGPUModel(),
                "machine": self.machineName,
                "os": "%s %s" % (u.system, u.release),
                "python": platform.python_version(),
                }


    def __getDefaultBenchDict(self, benchName):
        return {"params": [],
                "result": [],
                }

"""
benchmarks.json

{
    "<algo name>": {
        "code": "",
        "name": "<algo name>",
        "param_names": [
            "dataset_name",
        ],
        "params": [
            [
                "'dataset1'",
                "'dataset2'",
                "'dataset3'",
            ]
        ],
        "timeout": 0,
        "type": "time",
        "unit": "seconds",
        "version": 1,
    }
}
"""

"""
machine.json

{
    "arch": "x86_64",
    "cpu": "Intel, ...",
    "machine": "sm01",
    "os": "Linux ...",
    "ram": "123456",
    "version": 1,
}
"""

"""
<machine>-<commit_hash>.json

{
    "params": {
        "cuda": "9.2",
        "gpu": "Tesla ...",
        "machine": "sm01",
        "os": "Linux ...",
        "python": "3.7",
    },
    "requirements": {},
    "results": {
        "<algo name>": {
            "params": [
                [
                    "'dataset1'",
                    "'dataset2'",
                    "'dataset3'",
                ]
            ],
            "result": [
                <dataset1_exetime>,
                <dataset1_exetime>,
                <dataset1_exetime>,
            ],
        },
    },
    "commit_hash": "321e321321eaf",
    "date": 12345678,
    "python": "3.7",
    "version": 1,
}
"""

if __name__ == "__main__":
    asvDir = "asv"
    datasetName = "dolphins.csv"
    algoRunResults = [('loadDataFile', '3.2228727098554373 s'), ('createGraph', '0.00713360495865345 s'), ('pagerank', '0.00899268127977848 s'), ('bfs', '0.004273353144526482 s'), ('sssp', '0.004624705761671066 s'), ('jaccard', '0.0025573652237653732 s'), ('louvain', '0.32631026208400726 s'), ('weakly_connected_components', '0.0034315641969442368 s'), ('overlap', '0.002147899940609932 s'), ('triangles', '0.2544921860098839 s'), ('spectralBalancedCutClustering', '0.03329935669898987 s'), ('spectralModularityMaximizationClustering', '0.011258183047175407 s'), ('renumber', '0.001620553433895111 s'), ('view_adj_list', '0.000927431508898735 s'), ('degree', '0.0016251634806394577 s'), ('degrees', '0.002177216112613678 s')]
    cugraph_update_asv(asvDir, datasetName, algoRunResults)

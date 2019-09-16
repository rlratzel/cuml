import sys
#from time import process_time_ns   # only in 3.7!
from time import clock_gettime, CLOCK_MONOTONIC_RAW

class Nop:
    def __getattr__(self, attr):
        return Nop()
    def __getitem__(self, key):
        return Nop()
    def __call__(self, *args, **kwargs):
        return Nop()
nop = Nop()


# wrappers
def noStdoutWrapper(func):
    def wrapper(*args):
        prev = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            retVal = func(*args)
            sys.stdout = prev
            return retVal
        except Exception:
            sys.stdout = prev
            raise
    return wrapper


def logExeTime(func):
    def wrapper(*args):
        retVal = None
        perfData = logExeTime.perfData
        try:
            #st = process_time_ns()
            st = clock_gettime(CLOCK_MONOTONIC_RAW)
            retVal = func(*args)
        except Exception as e:
            perfData.append((func.__name__, "ERROR: %s" % e))
            return
        #exeTime = (process_time_ns() - st) / 1e9
        exeTime = clock_gettime(CLOCK_MONOTONIC_RAW) - st
        perfData.append((func.__name__, exeTime))
        return retVal
    return wrapper

logExeTime.perfData = []


def printLastResult(func):
    def wrapper(*args):
        retVal = func(*args)
        if printLastResult.perfData:
            (name, value) = printLastResult.perfData[-1]
            nameWidth = printLastResult.nameCellWidth
            valueWidth = printLastResult.valueCellWidth
            print("%s | %s" % (name.ljust(nameWidth),
                               str(value).ljust(valueWidth)))
        return retVal
    return wrapper

printLastResult.perfData = []
printLastResult.nameCellWidth = 40
printLastResult.valueCellWidth = 40


class WrappedFunc:
    wrappers = []

    def __init__(self, func, name="", args=None, extraRunWrappers=None):
        """
        func = the callable to wrap
        name = name of callable, needed mostly for bookkeeping
        args = args to pass the callable (default is no args)
        extraRunWrappers = list of functions that return a callable, used for
           wrapping the callable further to modify its environment, add timers,
           log calls, etc.
        """
        self.func = func
        self.name = name or func.__name__
        self.args = args or ()
        runWrappers = (extraRunWrappers or []) + self.wrappers

        # The callable is the callable obj returned after all wrappers applied
        for wrapper in runWrappers:
            self.func = wrapper(self.func)
            self.func.__name__ = self.name

    def run(self):
        return self.func(*self.args)


class Benchmark(WrappedFunc):
    wrappers = [logExeTime, printLastResult]

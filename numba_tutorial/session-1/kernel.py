import time

import numba
import numpy as np
from numba import cuda

from src.utils.tools import pyout


def cpu_histogram(x, xmin, xmax, histogram_out):
    """Increment bin counts in histogram_out, given histogram range [xmin, xmax)."""
    # Not that we don't have to pass in nbins explicitly, because the size of histogram_out determines it
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    for element in x:
        bin_number = np.int32((element - xmin) / bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            # only increment if in range
            histogram_out[bin_number] += 1


@cuda.jit
def gpu_histogram(x, xmin, xmax, histogram_out):
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.shape[0], stride):
        bin_number = numba.int32((x[i] - xmin) / bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            cuda.atomic.add(histogram_out, bin_number, 1)


x = np.random.normal(size=100000, loc=0, scale=1).astype(np.float32)
xmin = np.float32(-4.)
xmax = np.float32(4.)
for _ in range(10):
    t0 = time.time()
    histogram_out = np.zeros(shape=10, dtype=np.int32)
    cpu_histogram(x, xmin, xmax, histogram_out)
    # pyout(histogram_out)
    t1 = time.time()
    histogram_out = np.zeros(shape=10, dtype=np.int32)
    gpu_histogram[30, 128](x, xmin, xmax, histogram_out)
    # pyout(histogram_out)
    pyout(t1 - t0, time.time() - t1)

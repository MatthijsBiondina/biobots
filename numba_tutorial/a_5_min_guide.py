import numpy as np
from numba import cuda

from src.utils.tools import pyout

x = np.arange(10)
d_x = cuda.to_device(x)
x_ = d_x.copy_to_host()
pyout()


# @cuda.jit
# def increment_by_one(an_array: np.ndarray):
#     """
#     Increment all array elements by one
#     :param an_array:
#     :return:
#     """
#     pass
#
#
# an_array = np.arange(100)
#
# threadsperblock = 32
# blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
# increment_by_one[blockspergrid, threadsperblock](an_array)
#
# print()
# print()

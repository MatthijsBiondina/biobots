from torch import Tensor
import numpy as np

"""DEBUG watches"""

"""
from utils.cantrips import tensor_shapes, array_shapes
tensor_shapes([(loc_, eval(loc_)) for loc_ in locals()])
array_shapes([(loc_, eval(loc_)) for loc_ in locals()])

lcl = tensor_shapes({key: value for key, value in locals().items() if key != 'self'})
"""


def array_shapes(loc):
    loc = sorted(loc)
    ou = {}
    for name, value in loc:
        if isinstance(name, str) and name[0] == '_':
            continue
        if issubclass(type(value), np.ndarray):
            ou[name] = tuple(value.shape)
        elif isinstance(value, dict):
            ou[name] = array_shapes([(key, value[key]) for key in value])
        elif (isinstance(value, tuple) or isinstance(value, list)) \
                and any(issubclass(type(x), np.ndarray) for x in value):
            as_dict = array_shapes([(ii, value[ii]) for ii in range(len(value))])
            as_list = [None] * (max(as_dict.keys()) + 1)
            for ii in as_dict.keys():
                as_list[ii] = as_dict[ii]
            ou[name] = as_list

    return dict(sorted(ou.items()))


def tensor_shapes(loc, ignore_key='__lcl'):
    ou = {}
    for name in loc:
        value = loc[name]
        if isinstance(value, Tensor):
            ou[name] = tuple(value.size())
        elif isinstance(value, dict):
            ou[name] = tensor_shapes(value)
        # elif isinstance(value, list) or isinstance(value, tuple):
        #     loc_ = {ii: val for ii, val in enumerate(value)}
        #     ou[name] = tensor_shapes(loc_)
        else:
            try:
                loc_ = {key: val for key, val in value.__dict__.items()}
                ou[name] = tensor_shapes(loc_)
            except AttributeError:
                pass

    return {key: ou[key] for key in sorted(ou) if ou[key] != {} and key != ignore_key}


def as_numpy(loc, ignore_key='__lcl'):
    ou = {}
    for name in loc:
        value = loc[name]
        if isinstance(value, Tensor):
            ou[name] = value.numpy()
        elif isinstance(value, np.ndarray):
            ou[name] = value
        elif isinstance(value, dict):
            ou[name] = as_numpy(value)
        # elif isinstance(value, list) or isinstance(value, tuple):
        #     loc_ = {ii: val for ii, val in enumerate(value)}
        #     ou[name] = as_numpy(loc_)
        else:
            try:
                loc_ = {key: val for key, val in value.__dict__.items()}
                ou[name] = as_numpy(loc_)
            except AttributeError:
                pass

    return {key: ou[key] for key in sorted(ou) if len(ou[key]) and key != ignore_key}

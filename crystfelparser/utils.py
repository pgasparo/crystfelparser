"""Functions for mapping a dictionary to a h5 file and opposite"""

from collections import defaultdict
import numpy as np
import h5py


# ----- From dictionary to h5 and viceversa ------

def save_dict_to_hdf5(dic, filename):
    """
    Simple fuction to save a dictionary to a h5 file.

    Args:
      dic: the dictionary to be saved
      filename: the path where to save the dictionary
    """

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Recursively unwrap the dictionary contante to save it to h5

    Args:
      h5file: an open h5 file
      path: path within the h5 file
      dic: the dictionary to save
    """

    for key, item in dic.items():
        if isinstance(item, (np.ndarray, int, float, str, bytes)):
            h5file[path + str(key)] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(
                h5file, path + str(key) + '/', item)
        else:
            print(item)
            raise ValueError('Cannot save %s type' % type(item))


def load_dict_from_hdf5(filename):
    """
    Load an h5 file and cast it to a dictionary

    Args:
      filename: the path of the h5 to load

    Returns:
      A dictionary
    """
    with h5py.File(filename, 'r') as h5file:
        tmp_dictionary = recursively_load_dict_contents_from_group(h5file, '/')
        # cast the main keys from string to int
        return dict({int(k): v for k, v in tmp_dictionary.items()})


def recursively_load_dict_contents_from_group(h5file, path):
    """
    Recursively go through nested dictionaries

    Args:
      h5file: an open h5 file
      path: path within the h5 file

    Returns:
      A dictionary
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans

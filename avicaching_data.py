#!/usr/bin/env python

# Reads, writes, mutates and operates on avicaching data files. This module
# is imported in all models for reading and doing things.
# -----------------------------------------------------------------------------
# What do terms mean:
#   X - visit densities before placing rewards (for Identification Problem)
#   Y - visit densities after placing rewards (for Identification Problem)
#   R - rewards matrix
#   J - no. of locations
#   T - no. of time units

import numpy as np
import json
import csv
import os
from functools import reduce

def read_data_settings_file(fname):
    """
    Read the data settings file containing a JSON format of XYR, F, DIST,
    directory etc. information.

    Args:
        fname -- (str) name of the file

    Returns:
        2-tuple -- (tuple of 2 dicts) The first dictionary has information
            about the original data files. The second dictionary has
            information about the random data files.
    """
    DATA_JSON = json.load(open(fname, "r"))

    # orig data files' locations
    ORIG_DATA_FILES = {
        'XYR': DATA_JSON['orig']['XYR_file'],
        'F': DATA_JSON['orig']['F_file'],
        'DIST': DATA_JSON['orig']['DIST_file']
    }

    # rand data files' locations
    def rand_data_fname(key):
        """
        Returns the file name for rand data associated with key type of
        information.
        """
        return "{0:s}{1:d}{2:s}".format(
            DATA_JSON['rand'][key]['pre'],
            DATA_JSON['rand']['locs_in_file'],
            DATA_JSON['rand'][key]['suff']
        )

    RAND_DATA_FILES = {
        'XYR': rand_data_fname('XYR_file'),
        'F': rand_data_fname('F_file'),
        'DIST': rand_data_fname('DIST_file'),
        'XYR_weights': rand_data_fname('XYR_weights_file'),
    }

    return ORIG_DATA_FILES, RAND_DATA_FILES

def read_XYR_file(fname, locs, T, dtype=np.float32):
    """
    Read the datafile containing X, Y, R information.

    Args:
        fname -- (str) name of the file
        locs -- (int) J
        T -- (int) T
        dtype -- (np.dtype) data type for loading NumPy array

    Returns:
        3-tuple -- (tuple of NumPy ndarrays) X, Y, R
    """
    XYR = np.loadtxt(fname, dtype=dtype)[:T*3, :locs]  # only first locs cols
    # X Y R rows are interspersed, so index them at every 3 rows
    return (XYR[0::3, :], XYR[1::3, :], XYR[2::3, :])

def get_is_avi_vec(fname, locs, col):
    """
    Returns the is_avi integer vector from fname file indicating if locations
    are avicaching locations or not.

    Args:
        fname -- (str) name of the file containing environmental features
        locs -- (int) J
        col -- (int) column number of the file in which is_avi data is present
            (columns are 0 indexed); col is typically the last col of the file

    Returns:
        NumPy ndarray -- is_avi vector
    """
    return np.loadtxt(fname, dtype=np.int, usecols=col, skiprows=1)[:locs]

def read_F_file(fname, locs, dtype=np.float32):
    """
    Reads the csv file containing f information and is_avi indicators.

    Args:
        fname -- (str) name of the file
        locs -- (int) J
        dtype -- (np.dtype) data type for loading NumPy array

    Returns:
        NumPy ndarray -- f, is_avi
    """
    # first row of csv file is header row
    F_all = np.genfromtxt(fname, dtype=None, skip_header=1,
                          delimiter=",", encoding=None)[:locs]

    # genfromtxt outputs a list of tuples (a tuple for each row)
    F_all = list(map(list, F_all))

    # last 4 cols are lat-long info, loc_id, and is_avi indicator
    F = np.asarray([row[:-4] for row in F_all], dtype=dtype)
    # could use get_is_avi_vec but we already have F_all here
    is_avi = np.asarray([row[-1] for row in F_all], dtype=dtype)
    return (F, is_avi)

def read_dist_file(fname, locs, dtype=np.float32):
    """
    Reads the DIST file containing the distances between all locations.

    Args:
        fname -- (str) name of the file
        locs -- (int) J
        dtype -- (np.dtype) data type for loading NumPy array

    Returns:
        NumPy ndarray -- DIST
    """
    DIST = np.loadtxt(fname, dtype=dtype)[:locs, :locs]
    return DIST

def combine_DIST_F(f, DIST, locs, num_features, dtype=np.float32):
    """
    Combines f and DIST as data preprocessing.

    Args:
        F -- (NumPy ndarray) f
        DIST -- (NumPy ndarray) DIST
        locs -- (int) J
        num_features -- (int) `len(f[i]) + 1` (accounting for the distance element)
        dtype -- (np.dtype) data type for loading NumPy array

    Returns:
        NumPy ndarray -- represents the input dataset without the rewards.
    """
    NN_in = np.empty([locs, locs, num_features], dtype=dtype)
    for v in range(locs):
        for u in range(locs):
            NN_in[v][u][0] = DIST[v][u]
            NN_in[v][u][1:] = f[u]
    return NN_in

def save_rand_XYR(fname, X, Y, R, J=116, T=173):
    """
    Writes X, Y, R information to a file such that it is readable by read_XYR_file().
    The dimensions of X, Y, R must be J x T each.

    Args:
        fname -- (str) name of the file
        X -- (NumPy ndarray) X
        Y -- (NumPy ndarray) Y
        R -- (NumPy ndarray) R
        J -- (int) no. of locations (default=116)
        T -- (int) no. of time units (default=173)
    """
    # intersperse XYR
    XYR = np.empty([T * 3, J])
    XYR[0::3, :] = X    # every third row starting from row 0 is a row of X
    XYR[1::3, :] = Y
    XYR[2::3, :] = R
    np.savetxt(fname, XYR, fmt="%.8f", delimiter=" ")

def read_weights_file(fname, n_weights, locs_in_file, locs, num_features, dtype=np.float32):
    """
    Reads the weights file and splits the saved data into 2 weights tensors,
    as our models require.

    Args:
        fname -- (str) name of the file
        locs_in_file -- (int) no. of locations used in the weights file
        locs -- (int) no. of locations required by the run specs
            (<= locs_in_file). The function reads the file and ignores data
            corresponding to locations > locs
        num_features -- (int) no. of features in the dataset (the run spec
            should not request something different than what is in the file).
            This also determines the size of the weights tensors
        dtype -- (np.dtype) data type for loading NumPy array

    Returns:
        2-tuple of NumPy ndarrays -- w1 and w2
    """
    # first line contains specs
    data = np.loadtxt(fname, skiprows=1, dtype=dtype)
    # data is an ndarray with the 1st part representing the 3d w[0:-1] (represented as
    # 2d slices) and the 2nd part representing w[-1] - the last locs_in_file rows

    split_at_idx = [(i * locs_in_file * num_features) \
                    for i in range(1, n_weights)]
    w = np.split(data, split_at_idx, axis=0)

    # take out only locs slices. Since w[:-1] is represented in 2d, this means
    # taking out locs * num_features slices
    w[-1] = w[-1][:locs]
    w[:-1] = list(map(lambda wi: wi[:locs * num_features], w[:-1]))

    # reshape w[:-1]
    w[:-1] = list(map(
        lambda wi: wi.reshape((locs, num_features, num_features)), w[:-1]
    ))

    return w

def read_lat_long_from_Ffile(fname, locs, lat_col=33, long_col=34):
    """
    Reads the latitude and longitude from the file containing f information.

    Args:
        fname -- (str) name of the file
        locs -- (int) no. of locations. Also represents the length of lat and
            long vectors
        lat_col -- (int) col no. in the f file (default=33)
        long_col -- (int) col no. in the f file (default=34)

    Returns:
        NumPy ndarray -- 2d matrix where the first col are latitudes and the
        second col are longitudes
    """
    # skip header row of csv
    lat_long = np.loadtxt(fname, skiprows=1, delimiter=",", usecols=(
        lat_col, long_col))[:locs, :]
    return lat_long

def normalize(x, along_dim=None, using_max=True, offset_division=0.000001):
    """
    Normalizes a tensor by dividing each element by the maximum or by the sum,
    which are calculated along a dimension.

    Args:
        x -- (NumPy ndarray) matrix/tensor to be normalized
        along_dim -- (int or None) If along_dim is an int, the max is
            calculated along that dimension; if it's None,
            whole x's max/sum is calculated (default=None)
        using_max -- (bool) Normalize using max if True and sum if False
            (default=True)
        offset_division -- (float) safety mechanism to avoid division by zero
            (default=0.000001)

    Returns:
        NumPy ndarray -- Normalized matrix/tensor
    """
    if using_max:
        return x / (np.amax(x, axis=along_dim) + offset_division)
    # else
    return x / (np.sum(x, axis=along_dim, keepdims=True) + offset_division)

def make_rand_F_file(fname, J):
    """
    [Extremely bad code. A very bad example of coding style]
    Creates and write random f file.

    Args:
        fname -- (str) name of the file
        J -- (int) J
    """
    # num visits  -- type random int
    num_visits = np.floor(np.random.rand(J) * 1000)
    # num species -- type random int
    num_species = np.floor(np.random.rand(J) * 500)
    # NLCD2011_FS_C11_375_PLAND -- type random float
    NLCD2011_FS_C11_375_PLAND = np.random.rand(J) * 100
    # NLCD2011_FS_C12_375_PLAND -- zeros
    NLCD2011_FS_C12_375_PLAND = np.zeros(J)
    # NLCD2011_FS_C21_375_PLAND -- type random float
    NLCD2011_FS_C21_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C22_375_PLAND -- type random float
    NLCD2011_FS_C22_375_PLAND = np.random.rand(J) * 50
    # NLCD2011_FS_C23_375_PLAND -- type random float
    NLCD2011_FS_C23_375_PLAND = np.random.rand(J) * 50
    # NLCD2011_FS_C24_375_PLAND -- type random float
    NLCD2011_FS_C24_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C31_375_PLAND -- type random float
    NLCD2011_FS_C31_375_PLAND = np.random.rand(J) * 2
    # NLCD2011_FS_C41_375_PLAND -- type random float
    NLCD2011_FS_C41_375_PLAND = np.random.rand(J) * 100
    # NLCD2011_FS_C42_375_PLAND -- type random float
    NLCD2011_FS_C42_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C43_375_PLAND -- type random float
    NLCD2011_FS_C43_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C52_375_PLAND -- type random float
    NLCD2011_FS_C52_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C71_375_PLAND -- type random float
    NLCD2011_FS_C71_375_PLAND = np.random.rand(J) * 2
    # NLCD2011_FS_C81_375_PLAND -- type random float
    NLCD2011_FS_C81_375_PLAND = np.random.rand(J) * 100
    # NLCD2011_FS_C82_375_PLAND -- type random float
    NLCD2011_FS_C82_375_PLAND = np.random.rand(J) * 80
    # NLCD2011_FS_C90_375_PLAND -- type random float
    NLCD2011_FS_C90_375_PLAND = np.random.rand(J) * 20
    # NLCD2011_FS_C95_375_PLAND -- type random float
    NLCD2011_FS_C95_375_PLAND = np.random.rand(J) * 2
    # HOUSING_DENSITY -- type random float
    HOUSING_DENSITY = np.random.rand(J) * 500
    # HOUSING_PERCENT_VACANT -- type random float
    HOUSING_PERCENT_VACANT = np.random.rand(J) * 0.1
    # ELEV_GT -- type random int
    ELEV_GT = np.floor(np.random.rand(J) * 500)
    # DIST_FROM_FLOWING_FRESH -- type random int
    DIST_FROM_FLOWING_FRESH = np.floor(np.random.rand(J) * 5)
    # DIST_IN_FLOWING_FRESH -- type random int
    DIST_IN_FLOWING_FRESH = np.floor(np.random.rand(J) * 10)
    # DIST_FROM_STANDING_FRESH -- type random int
    DIST_FROM_STANDING_FRESH = np.floor(np.random.rand(J) * 10)
    # DIST_IN_STANDING_FRESH -- type random int
    DIST_IN_STANDING_FRESH = np.floor(np.random.rand(J) * 10)
    # DIST_FROM_WET_VEG_FRESH -- type random int
    DIST_FROM_WET_VEG_FRESH = np.floor(np.random.rand(J) * 10)
    # DIST_IN_WET_VEG_FRESH -- type random int
    DIST_IN_WET_VEG_FRESH = np.floor(np.random.rand(J) * 10)
    # DIST_FROM_FLOWING_BRACKISH -- type random int
    DIST_FROM_FLOWING_BRACKISH = np.floor(np.random.rand(J) * 10)
    # DIST_IN_FLOWING_BRACKISH -- type random int
    DIST_IN_FLOWING_BRACKISH = np.floor(np.random.rand(J) * 10)
    # DIST_FROM_STANDING_BRACKISH -- type random int
    DIST_FROM_STANDING_BRACKISH = np.floor(np.random.rand(J) * 10)
    # DIST_IN_STANDING_BRACKISH -- type random int
    DIST_IN_STANDING_BRACKISH = np.floor(np.random.rand(J) * 10)
    # DIST_FROM_WET_VEG_BRACKISH -- type random int
    DIST_FROM_WET_VEG_BRACKISH = np.floor(np.random.rand(J) * 10)
    # DIST_IN_WET_VEG_BRACKISH -- type random int
    DIST_IN_WET_VEG_BRACKISH = np.floor(np.random.rand(J) * 10)
    # LATITUDE -- type intersperse between 42 44
    LATITUDE = np.linspace(42, 44, num=J)
    # LONGITUDE --type intersperse between -75 -77
    LONGITUDE = np.linspace(-75, -77, num=J)
    # LOC_ID -- type random
    LOC_ID = np.random.rand(J)
    # IS_AVI_LOC -- type random int, rounded
    IS_AVI_LOC = np.random.rand(J)
    IS_AVI_LOC[IS_AVI_LOC < 0.5] = 0
    IS_AVI_LOC[IS_AVI_LOC >= 0.5] = 1

    ###
    data = np.vstack([num_visits,
                      num_species,
                      NLCD2011_FS_C11_375_PLAND,
                      NLCD2011_FS_C12_375_PLAND,
                      NLCD2011_FS_C21_375_PLAND,
                      NLCD2011_FS_C22_375_PLAND,
                      NLCD2011_FS_C23_375_PLAND,
                      NLCD2011_FS_C24_375_PLAND,
                      NLCD2011_FS_C31_375_PLAND,
                      NLCD2011_FS_C41_375_PLAND,
                      NLCD2011_FS_C42_375_PLAND,
                      NLCD2011_FS_C43_375_PLAND,
                      NLCD2011_FS_C52_375_PLAND,
                      NLCD2011_FS_C71_375_PLAND,
                      NLCD2011_FS_C81_375_PLAND,
                      NLCD2011_FS_C82_375_PLAND,
                      NLCD2011_FS_C90_375_PLAND,
                      NLCD2011_FS_C95_375_PLAND,
                      HOUSING_DENSITY,
                      HOUSING_PERCENT_VACANT,
                      ELEV_GT,
                      DIST_FROM_FLOWING_FRESH,
                      DIST_IN_FLOWING_FRESH,
                      DIST_FROM_STANDING_FRESH,
                      DIST_IN_STANDING_FRESH,
                      DIST_FROM_WET_VEG_FRESH,
                      DIST_IN_WET_VEG_FRESH,
                      DIST_FROM_FLOWING_BRACKISH,
                      DIST_IN_FLOWING_BRACKISH,
                      DIST_FROM_STANDING_BRACKISH,
                      DIST_IN_STANDING_BRACKISH,
                      DIST_FROM_WET_VEG_BRACKISH,
                      DIST_IN_WET_VEG_BRACKISH,
                      LATITUDE,
                      LONGITUDE,
                      LOC_ID,
                      IS_AVI_LOC])

    with open(fname, "w") as f:
        f.write("num visits,"
                "num species,"
                "NLCD2011_FS_C11_375_PLAND,"
                "NLCD2011_FS_C12_375_PLAND,"
                "NLCD2011_FS_C21_375_PLAND,"
                "NLCD2011_FS_C22_375_PLAND,"
                "NLCD2011_FS_C23_375_PLAND,"
                "NLCD2011_FS_C24_375_PLAND,"
                "NLCD2011_FS_C31_375_PLAND,"
                "NLCD2011_FS_C41_375_PLAND,"
                "NLCD2011_FS_C42_375_PLAND,"
                "NLCD2011_FS_C43_375_PLAND,"
                "NLCD2011_FS_C52_375_PLAND,"
                "NLCD2011_FS_C71_375_PLAND,"
                "NLCD2011_FS_C81_375_PLAND,"
                "NLCD2011_FS_C82_375_PLAND,"
                "NLCD2011_FS_C90_375_PLAND,"
                "NLCD2011_FS_C95_375_PLAND,"
                "HOUSING_DENSITY,"
                "HOUSING_PERCENT_VACANT,"
                "ELEV_GT,"
                "DIST_FROM_FLOWING_FRESH,"
                "DIST_IN_FLOWING_FRESH,"
                "DIST_FROM_STANDING_FRESH,"
                "DIST_IN_STANDING_FRESH,"
                "DIST_FROM_WET_VEG_FRESH,"
                "DIST_IN_WET_VEG_FRESH,"
                "DIST_FROM_FLOWING_BRACKISH,"
                "DIST_IN_FLOWING_BRACKISH,"
                "DIST_FROM_STANDING_BRACKISH,"
                "DIST_IN_STANDING_BRACKISH,"
                "DIST_FROM_WET_VEG_BRACKISH,"
                "DIST_IN_WET_VEG_BRACKISH,"
                "LATITUDE,"
                "LONGITUDE,"
                "LOC_ID,"
                "IS_AVI_LOC\n")
        np.savetxt(f, data.T, fmt="%.5f", delimiter=",")

def make_rand_DIST_file(fname, J):
    """
    Creates random DIST file. J x J random matrix max 100. diagonal elements 0

    Args:
        fname -- (str) name of the file
        J -- (int) J
    """
    data = np.random.rand(J, J) * 100
    data[np.diag_indices(J)] = 0.0  # distance[u][u] = 0
    np.savetxt(fname, data, fmt="%.6f", delimiter=" ")

def expand_vec_with_is_avi(vec_to_exp, is_avi, locs_avi, locs, dtype=np.float32):
    """
    Expands vec_to_exp (length locs_avi) to a vector of length locs determined
    by the entries of is_avi (length locs). is_avi is a vector of ones and zeros,
    ones at indices representing avicaching locations. Lowest i s.t. is_avi[i] ==
    1 means that the returned vector[i] will have vec_to_exp[0]. The next i with
    this property will mean that the returned vector[i] will have vec_to_exp[1].
    And so on.

    Args:
        vec_to_exp -- (NumPy ndarray) vector of len locs_avi to be expanded into
            a vector of length locs
        is_avi -- (NumPy ndarray) vector of len locs representing where the
            avicaching locations are; zero entries mean non-avicaching locations,
            one entries mean avicaching locations
        locs_avi -- (int) number of avicaching locations; is_avi must have exactly
            locs_avi number of ones and rest must be zeros
        locs -- (int) length of is_avi vector
        dtype -- (np.dtype) data type for loading NumPy array

    Returns:
        NumPy ndarray -- expanded vector
    """

    vec_exp = np.zeros(locs, dtype=dtype)
    num_avi_loc_enc = 0     # no. of avicaching locs encountered in vec_to_exp
    for i in range(locs):
        if np.abs(is_avi[i] - 1) < 1e-10:
            # i is an index for an avicaching loc
            vec_exp[i] = vec_to_exp[num_avi_loc_enc]
            num_avi_loc_enc += 1

    return vec_exp

def distribute_save_rewards(f_gen_rewards, save_fname, totalR, zero_at_avi=True):
    """
    Generates rewards using function f_gen_rewards and saves in save_fname file.

    After distributing rewards in all locations, the distribution is normalized
    to 1 using the sum of rewards and multiplied by totalR thereafter.

    This ensures that rewards at all locations sum to totalR. Then, zero
    rewards are placed at non-avicaching locations if zero_at_avi is True.
    Note that this might mean that the resulting rewards do not sum to totalR.

    Args:
        f_gen_rewards -- (lambda X, X_avi) Returns either a len(X) or a
            len(X_avi) vector of rewards distributed using either X or X_avi
            vector. X are the visit densities for all locations before the
            rewards are placed, and X_avi are the visit densities only for
            avicaching locations extracted from X.
        save_fname -- (str) file name for saving
        totalR -- (int) total rewards that should be distributed combining all
            locations

    Examples:
        1. To distribute random rewards totalling 1000, at avicaching locations,
        we can do this:

        ```python
        distribute_save_rewards(
            lambda X, X_avi: np.random.rand(len(X_avi),
            "saved_rewards.txt",
            1000
        )
        ```

        2. To distribute equal rewards at all locations totalling 500 rewards,
        we can do the following:

        ```python
        distribute_save_rewards(
            lambda X, X_avi: np.ones(len(X)),
            "saved_rewards.txt",
            500,
            zero_at_avi=False
        )
        ```
    """
    J = 116
    T = 182

    # Read X, is_avi data
    ORIG_DATA_FILES, _ = read_data_settings_file(
        "./nn_avicaching_data_settings.json"
    )
    X, _, _ = read_XYR_file(ORIG_DATA_FILES['XYR'], J, T)
    F, is_avi = read_F_file(ORIG_DATA_FILES['F'], J)
    X = normalize(X.sum(axis=0), using_max=False)

    # Find indices of avi locations, extract X for these locations
    is_avi_ind = np.nonzero(is_avi) # nonzero elts represent avi locations
    J_avi = len(is_avi_ind[0])  # is_avi_ind is a tuple, calc num of indices
    X_avi = X[is_avi_ind]

    # Calculate rewards, expand rewards, store
    r = totalR * normalize(f_gen_rewards(X, X_avi), using_max=False)
    if zero_at_avi:
        r = expand_vec_with_is_avi(r, is_avi, J_avi, J)
    np.savetxt(save_fname, np.expand_dims(r, axis=0), fmt="%.10f")

def generate_proportional_rewards(save_fname, totalR):
    """
    Generates rewards inversely proportional to the relative number of
    visits to each avicaching location in visit densities before the rewards
    were placed. Zero rewards are placed at non-avicaching locations.

    Args:
        save_fname -- (str) file name for saving
        totalR -- (int) total rewards that should be distributed combining all
            locations
    """
    distribute_save_rewards(
        lambda X, X_avi: 1 / (X_avi + 1e-6),
        save_fname,
        totalR
    )

def generate_random_rewards(save_fname, totalR):
    """
    Generates random rewards on the avicaching locations such that the rewards
    sum to totalR. Zero rewards are placed at non-avicaching locations.

    Args:
        save_fname -- (str) file name for saving
        totalR -- (int) total rewards that should be distributed combining all
            locations.
    """
    distribute_save_rewards(
        lambda X, X_avi: np.random.rand(len(X_avi)),
        save_fname,
        totalR
    )

def generate_equal_rewards(save_fname, totalR):
    """
    Generates equal rewards on the avicaching locations such that the rewards
    sum to totalR. Zero rewards are placed at non-avicaching locations.

    Args:
        save_fname -- (str) file name for saving
        totalR -- (int) total rewards that should be distributed combining all
            locations.
    """
    distribute_save_rewards(
        lambda X, X_avi: np.ones(len(X_avi)),
        save_fname,
        totalR
    )

def randint_upto_sum(sum_randints, n):
    """
    Generates a list of n integers that sum up to sum_randints.

    Args:
        sum_randints -- (int) Sum of integers
        n -- (int) Number of integers
    """
    # think of sum_randints on the number line. Divide the line from 0 to n
    # into n segments by placing n-1 random marks, and find the sizes of them.
    line_marks = [0] + sorted(
        [np.random.randint(0, sum_randints+1) for _ in range(n-1)]
    ) + [sum_randints]
    seg_sizes = []
    for i in range(n):
        seg_sizes.append(line_marks[i+1] - line_marks[i])
    return seg_sizes

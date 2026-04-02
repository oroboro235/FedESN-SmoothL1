# this file is for searching parameters for each dataset and each model
# 

import json
import time
import os

import random
import numpy as np
np.random.seed(1234)

from reservoirpy.nodes import Reservoir, Ridge
from readout_node import Reg_Node, Clr_Node

from data_loader import read_data, uni_names
from config import path

import scipy.io
from functools import partial
import multiprocessing
from multiprocessing import Pool

# datasets
# Reg datasets:
#   - Mackey-Glass
#   - Lorenz64
# Clr datasets:
#   - HAR
#   - Char
#   - Distal
#   - ECG5000
#   - Strawberry
#   - Yoga
#   - JapaneseVowel


tsr_datasets_name = [
    "mg",
    "lorenz"
]

tsc_datasets_name = [
    "jpv",
    "har",
    "char",
] + uni_names

# models
# - None (only Clr)
# - L2
# - L1
# - SL1 (SmoothL1)

reg_types = [
    "none",
    "l2",
    "l1",
    "sl1"
]

# hyper-parametes to search
# - Reservoir:
#   - spectral radius (sr)          (0.0 - 5.0]
#   - leaking rate (gamma)          (0.0 - 1.0]
#   - input scaling (scale_i)       [1e-2, 1e-1, 1e0, 1e1, 1e2]
#   - regularization (reg_param)    [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
# searching method - Random Search



def randomSearch(X, y, len_valid, k=1, steps=100,
                 params_forSearch_range={}, params_fixed={},
                 metric="mse"):
    



def main():
    # regression tasks
    for dataset_name in tsr_datasets_name:
        for reg_type in reg_types:
            if reg_type == "none" and dataset_name in ["mg", "lorenz"]:
                continue
            result = {}
            X, y, _, _ = read_data(dataset_name)





            

    
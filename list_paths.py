import os
import pandas as pd
import anndata as ad
import re
import numpy as np


def create_list_of_paths(
    directory, stimulation, cell_type=None, sample=None, patient_excluded=None
):
    paths_list = []
    for filename in os.listdir(directory):
        if (
            stimulation in filename
            and (cell_type is None or cell_type in filename)
            and (sample is None or sample in filename)
            and (patient_excluded is None or patient_excluded not in filename)
        ):
            paths_list.append(os.path.join(directory, filename))
    return paths_list


def create_list_of_paths_spec_patients(
    directory, stimulation, cell_type=None, sample=None, patient=None
):
    paths_list = []
    for filename in os.listdir(directory):
        if (
            stimulation in filename
            and (cell_type is None or cell_type in filename)
            and (sample is None or sample in filename)
            and (patient is None or patient in filename)
        ):
            paths_list.append(os.path.join(directory, filename))
    return paths_list


def count_files(directory, stimulation):
    count = 0
    sumsize = 0
    for filename in os.listdir(directory):
        if filename.endswith(".fcs") and stimulation in filename:
            count += 1
            sumsize += os.path.getsize(os.path.join(directory, filename))
    return count, sumsize

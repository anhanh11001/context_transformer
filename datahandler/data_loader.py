import os

import pandas

from utils import print_line_divider
from numpy import vstack


def load_date_from_steptracking_file(filepath, show_info=False):
    data = pandas.read_csv(filepath)
    data['date'] = pandas.to_datetime(data['date'], format='%d %b %Y %H:%M:%S:%f %z')
    dates = []
    for value in data.values[:, 0]:
        dates.append(value.to_pydatetime())
    return dates


def load_data_from_file(filepath, show_info=False):
    data = pandas.read_csv(filepath)
    data['date'] = pandas.to_datetime(data['date'], format='%d %b %Y %H:%M:%S:%f %z')
    data = data.set_index('date')
    if show_info:
        print("Information about the panda file: Shape " + str(data.shape) + " and some header fields:")
        print(data.head())
        print_line_divider()
    return data


def get_files_from_folder(folder):
    files = []
    for path, subdir, filenames in os.walk(folder):
        for filename in filenames:
            if filename.startswith("."):
                continue
            files.append(os.path.join(path, filename))
    return files


def load_data_from_files(filepaths, show_info=False):
    loaded = list()
    for filepath in filepaths:
        df = load_data_from_file(filepath)
        if show_info:
            print("Loaded data from file " + filepath + " with shape: " + str(df.shape))
        loaded.append(df)
    for f in loaded:
        print(f.shape)
    return vstack(loaded)


def load_data_from_folder(foldername, show_info=False):
    if show_info:
        print("Loading data from folder " + foldername + "...")
    files = get_files_from_folder(foldername)
    loaded = load_data_from_files(files)
    if show_info:
        print("Loaded data with shape " + str(loaded))
    return loaded

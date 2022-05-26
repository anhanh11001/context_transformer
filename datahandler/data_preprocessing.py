import os
import random
from datetime import timedelta

import numpy as np

from datahandler.constants import *
from datahandler.data_visualiser import load_data_from_file
from random import random


# NORMALIZATION
def normalize_series(series):
    max_series = max(series)
    min_series = min(series)
    diff = max_series - min_series
    normalized_series = (series - min_series) / diff
    return normalized_series


def normalize_data(df):
    for feature in all_features:
        df[feature] = normalize_series(df[feature])
    return df


# SLICING INTO WINDOW OF DATA
def convert_data_into_fixed_window_np(
        data,
        features,
        window_time_in_seconds=1,
        window_size=16,
):
    # For default input, we would have 1-second window with 8*n data points (n - # of features)
    # Input would be a dataframe and output would be a list of data with some fixed_window_size
    converted_data_list = []
    converted_label_list = []

    # Get the initial timestamp
    current_timestamp_converted = data.index[0].to_pydatetime()
    last_timestamp_raw = data.index[data.shape[0] - 1].to_pydatetime()
    current_timestamp_raw_index = 0
    window_length_in_milliseconds = window_time_in_seconds * 1000 / window_size

    while True:
        current_timestamp_converted = current_timestamp_converted + timedelta(
            milliseconds=window_length_in_milliseconds)
        if current_timestamp_converted > last_timestamp_raw:
            break

        while current_timestamp_raw_index < data.shape[0] - 1:
            next_timestamp_raw = data.index[current_timestamp_raw_index + 1].to_pydatetime()
            if next_timestamp_raw < current_timestamp_converted:
                current_timestamp_raw_index += 1
            else:
                break

        row_data = data.iloc[current_timestamp_raw_index]
        converted_data_list.append([row_data[att] for att in features])
        converted_label_list.append(location_labels.index(row_data[phone_label]))

    finalised_data_list = []
    finalised_label_list = []

    i = 0
    while window_size * (i / 2 + 1) < len(converted_data_list):
        start_ind = window_size * (i / 2)
        end_ind = window_size * (i / 2 + 1)
        finalised_data_list.append(np.array(converted_data_list[int(start_ind):int(end_ind)]))
        finalised_label_list.append(converted_label_list[int(start_ind)])
        i += 1
    return finalised_data_list, finalised_label_list


# FINALISED LOADING
def load_data(
        folder_name,
        features,
        window_time_in_seconds=1,
        window_size=16,
):
    collected_data = []
    collected_labels = []
    for label in os.listdir(folder_name):
        if label.startswith("."):
            continue
        label_dir = os.path.join(folder_name, label)
        if os.path.isdir(label_dir):
            for data_file in os.listdir(label_dir):
                if data_file.startswith("."):
                    continue
                filepath = os.path.join(label_dir, data_file)
                df = load_data_from_file(filepath)
                normalized = normalize_data(df)
                sub_collected_data, sub_collected_labels = convert_data_into_fixed_window_np(
                    normalized,
                    features,
                    window_time_in_seconds,
                    window_size
                )
                collected_data = collected_data + sub_collected_data
                collected_labels = collected_labels + sub_collected_labels
    return np.array(collected_data), np.array(collected_labels)


def get_train_test_data(features, window_time_in_seconds=1, window_size=16):
    train_x, train_y = load_data(train_folder, features, window_time_in_seconds, window_size)
    test_x, test_y = load_data(test_folder, features, window_time_in_seconds, window_size)
    return train_x, train_y, test_x, test_y

# data_from_csv = load_data_from_file(test_data_file)
# data, labels = convert_data_into_fixed_window_np(data_from_csv, 4, 80)
# data = np.array(data)
# labels = np.array(labels)
# print("Data shape: " + str(data.shape))
# print("First data shape: " + str(data[0].shape))
# print("Head data for label " + str(labels[0]) + ": " + str(data[0]))


def load_data_v3(folder=train_folder, features=all_features, window_time_in_seconds=1, window_size=32):
    all_data = []
    all_labels = []

    for datafile in os.listdir(folder):
        if datafile.startswith("."):
            continue
        print("Loading data from file " + datafile)
        filepath = os.path.join(folder, datafile)
        df = load_data_from_file(filepath)
        normalized = normalize_data(df)

        # Converting window for all data
        sub_all_data, sub_all_labels = convert_data_into_fixed_window_np(
            normalized,
            features,
            window_time_in_seconds,
            window_size
        )
        all_data = all_data + sub_all_data
        all_labels = all_labels + sub_all_labels

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    test_split = 0.1
    for i in range(len(all_data)):
        if random() < test_split: # Go to test set
            test_data.append(all_data[i])
            test_labels.append(all_labels[i])
        else:
            train_data.append(all_data[i])
            train_labels.append(all_labels[i])

    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

# train_x, train_y, test_x, test_y = load_data_v3(train_folder, all_features, window_time_in_seconds=1, window_size=32)
# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
# print(train_x)
# print(test_x)

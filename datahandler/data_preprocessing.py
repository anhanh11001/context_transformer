import os

import numpy as np

from datahandler.data_handler import train_folder, load_data_from_file
from datahandler.constants import *
from datetime import timedelta

test_data_file = train_folder + "/hand_holding/holdinginhand_data_0aff7db2-582f-4f08-b5d9-1f4742e0eb37.csv"
supported_features = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]


def convert_data_into_fixed_window_np(
        data,
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
        converted_data_list.append([row_data[att] for att in supported_features])
        converted_label_list.append(row_data[phone_label])

    finalised_data_list = []
    finalised_label_list = []
    for i in range(int(len(converted_data_list) / window_size)):
        finalised_data_list.append(np.array(converted_data_list[i * window_size:(i + 1) * window_size]))
        finalised_label_list.append(converted_label_list[i * window_size])

    return finalised_data_list, finalised_label_list


def load_all_data(folder_name):
    collected_data = []
    collected_labels = []
    for label in os.listdir(folder_name):
        if label.startswith("."):
            continue
        label_dir = os.path.join(folder_name, label)
        print("Transforming data from label folder: " + label + "...")
        if os.path.isdir(label_dir):
            for data_file in os.listdir(label_dir):
                if data_file.startswith("."):
                    continue
                filepath = os.path.join(label_dir, data_file)
                df = load_data_from_file(filepath)
                sub_collected_data, sub_collected_labels = convert_data_into_fixed_window_np(df)
                collected_data = collected_data + sub_collected_data
                collected_labels = collected_labels + sub_collected_labels
    return np.array(collected_data), np.array(collected_labels)


data, labels = load_all_data(train_folder)
print("Data shape: " + str(data.shape))
print("First data shape: " + str(data[0].shape))
print("Head data for label " + str(labels[0]) + ": " + str(data[0]))

# data_from_csv = load_data_from_file(test_data_file)
# data, labels = convert_data_into_fixed_window_np(data_from_csv)

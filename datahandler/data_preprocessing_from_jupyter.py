import math
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datahandler.constants import all_features, train_folder, location_labels
from datahandler.data_loader import load_data_from_file
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.decomposition import PCA
from random import random

DF_TYPE_RAW = 0
DF_TYPE_TIME_SERIES_DOMAIN_NORMALIZED = 2
DF_TYPE_TIME_SERIES_DOMAIN_PCA = 3
WINDOW_SIZE = 40
WINDOW_LENGTH_IN_SECONDS = 2
raw_features = all_features
added_features = ["accMag", "gyroMag", "magMag", "accAng", "gyroAng", "magAng"]
test_split = 0.2


def standardize_pca_df(df):
    feature_count = df.shape[1] - 1
    values = df.iloc[:, 0:feature_count]
    labels = df.loc[:, "labelPhone"].set_axis(range(df.shape[0]))
    standard_scaler = StandardScaler()
    standardized_values = standard_scaler.fit_transform(values)
    pca = PCA()
    pca_values = pca.fit_transform(standardized_values)
    pca_df = pd.DataFrame(data=pca_values, columns=values.columns)
    pca_df["labelPhone"] = labels

    return pca_df


def normalize_df(df):
    feature_count = df.shape[1] - 1
    values = df.iloc[:, 0:feature_count]
    labels = df.loc[:, "labelPhone"].set_axis(range(df.shape[0]))
    normalizer = MinMaxScaler()
    normalized_values = normalizer.fit_transform(values)
    normalized_df = pd.DataFrame(
        data=normalized_values,
        columns=values.columns
    )
    normalized_df["labelPhone"] = labels
    return normalized_df


def load_df_from_files(filepath, df_type):
    # Step 1: Load from data
    df = load_data_from_file(filepath)
    df = df.drop("labelActivity", axis=1)
    df['labelPhone'] = df['labelPhone'].apply(lambda x: location_labels.index(x))

    # Step 2: Divide collected data into fixed-size chunk
    fixed_size_data = []
    fixed_size_indexes = []
    current_timestamp = df.index[0].to_pydatetime()
    last_timestamp_raw = df.index[df.shape[0] - 1].to_pydatetime()
    current_timestamp_raw_index = 0
    one_window_length_in_millis = WINDOW_LENGTH_IN_SECONDS * 1000 / WINDOW_SIZE
    while True:
        current_timestamp = current_timestamp + timedelta(milliseconds=one_window_length_in_millis)
        if current_timestamp > last_timestamp_raw:
            break

        while current_timestamp_raw_index < df.shape[0] - 1:
            next_timestamp_raw = df.index[current_timestamp_raw_index + 1].to_pydatetime()
            if next_timestamp_raw < current_timestamp:
                current_timestamp_raw_index += 1
            else:
                break

        fixed_size_data.append(df.iloc[current_timestamp_raw_index])
        fixed_size_indexes.append(current_timestamp)
    fixed_size_df = pd.DataFrame(
        data=fixed_size_data,
        index=fixed_size_indexes,
        columns=df.columns
    )
    if df_type == DF_TYPE_RAW:
        return normalize_df(fixed_size_df)

    # Step 3: Added features
    accMag = []
    gyroMag = []
    magMag = []
    for index, row in fixed_size_df.iterrows():
        accMag.append(math.sqrt(row['accelerometerX'] ** 2 + row['accelerometerY'] ** 2 + row['accelerometerZ'] ** 2))
        gyroMag.append(math.sqrt(row['gyroscopeX'] ** 2 + row['gyroscopeY'] ** 2 + row['gyroscopeZ'] ** 2))
        magMag.append(math.sqrt(row['magnetometerX'] ** 2 + row['magnetometerY'] ** 2 + row['magnetometerZ'] ** 2))
    fixed_size_df["accMag"] = accMag
    fixed_size_df["gyroMag"] = gyroMag
    fixed_size_df["magMag"] = magMag

    def calculate_angle(input_x, input_y, input_z):
        dividend = input_x * 1 + input_y * 1 + input_z * 1
        divisor = math.sqrt(1 + 1 + 1) * math.sqrt(input_x ** 2 + input_y ** 2 + input_z ** 2)
        return math.acos(dividend / divisor)

    accAng = []
    gyroAng = []
    magAng = []
    for index, row in fixed_size_df.iterrows():
        accAng.append(calculate_angle(row["accelerometerX"], row["accelerometerY"], row["accelerometerZ"]))
        gyroAng.append(calculate_angle(row['gyroscopeX'], row['gyroscopeY'], row['gyroscopeZ']))
        magAng.append(calculate_angle(row["magnetometerX"], row["magnetometerY"], row["magnetometerZ"]))
    fixed_size_df['accAng'] = accAng
    fixed_size_df['gyroAng'] = gyroAng
    fixed_size_df['magAng'] = magAng

    # Step 4: Convert current features into time-series domain feature
    window_index_start = 0
    window_index_increasing_size = int(WINDOW_SIZE / 2)
    feature_columns = []
    feature_data = []

    domain_types = raw_features + added_features
    feature_types = ["mean", "std", "min", "max"]
    for domain_type in domain_types:
        for feature_type in feature_types:
            feature_columns.append(feature_type + domain_type)

    feature_columns.append("labelPhone")

    while window_index_start + WINDOW_SIZE < fixed_size_df.shape[0]:
        # Iteration per large sliding window of 20 windows.
        first_index = window_index_start
        last_index = window_index_start + WINDOW_SIZE
        sub_df = fixed_size_df[first_index:last_index]

        # Feature extraction from mean/max/min/std
        feature = []
        for domain_type in domain_types:
            raw_feature_series_describe = sub_df[domain_type].describe()
            for feature_type in feature_types:
                feature.append(raw_feature_series_describe[feature_type])
        feature_data.append(feature)

        # Final: Label
        feature.append(sub_df["labelPhone"][0])
        window_index_start += window_index_increasing_size

    features_df = pd.DataFrame(
        data=feature_data,
        columns=feature_columns
    )
    if df_type == DF_TYPE_TIME_SERIES_DOMAIN_NORMALIZED:
        return normalize_df(features_df)

    if df_type == DF_TYPE_TIME_SERIES_DOMAIN_PCA:
        return standardize_pca_df(features_df)

    raise Exception("Invalid DF type")


######### FINAL FUNCTIONS TO USE - Should return train/test set
def get_all_filepaths():
    res = []
    for datafile in os.listdir(train_folder):
        if datafile.startswith("."):
            continue
        res.append(os.path.join(train_folder, datafile))
    return res
    # return [test_data_file_v3]


def load_train_test_data_raw_normalized():
    train_x, train_y, test_x, test_y = [], [], [], []

    filepaths = get_all_filepaths()
    for i in range(len(filepaths)):
        filepath = filepaths[i]
        print("Loading from file: " + filepath + " (" + str(i + 1) + "/" + str(len(filepaths)) + ")")
        df = load_df_from_files(filepath, DF_TYPE_RAW)
        window_index_start = 0
        window_index_increasing_size = int(WINDOW_SIZE / 2)
        no_features = df.shape[1] - 1
        values = df.iloc[:, 0:no_features]
        labels = df.loc[:, "labelPhone"]
        while window_index_start + WINDOW_SIZE < df.shape[0]:
            if random() < test_split:
                test_x.append(values[window_index_start:(window_index_start + WINDOW_SIZE)])
                test_y.append(labels[window_index_start])
            else:
                train_x.append(values[window_index_start:(window_index_start + WINDOW_SIZE)])
                train_y.append(labels[window_index_start])

            window_index_start += window_index_increasing_size

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


def load_train_test_data_added_features_normalized():
    train_x, train_y, test_x, test_y = [], [], [], []

    filepaths = get_all_filepaths()
    for i in range(len(filepaths)):
        filepath = filepaths[i]
        print("Loading from file: " + filepath + " (" + str(i + 1) + "/" + str(len(filepaths)) + ")")
        df = load_df_from_files(filepath, DF_TYPE_TIME_SERIES_DOMAIN_NORMALIZED)
        no_features = df.shape[1] - 1
        values = df.iloc[:, 0:no_features]
        labels = df.loc[:, "labelPhone"]

        # window_index_start = 0
        # window_index_increasing_size = int(WINDOW_SIZE / 2)
        # while window_index_start + WINDOW_SIZE < df.shape[0]:
        #     if random() < test_split:
        #         test_x.append(values.iloc[i])
        #         test_y.append(labels[i])
        #     else:
        #         train_x.append(values.iloc[i])
        #         train_y.append(labels[i])
        #     window_index_start += window_index_increasing_size
        for i in range(df.shape[0]):
            if random() < test_split:
                test_x.append(values.iloc[i])
                test_y.append(labels[i])
            else:
                train_x.append(values.iloc[i])
                train_y.append(labels[i])
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


def load_train_test_data_added_features_pca():
    train_x, train_y, test_x, test_y = [], [], [], []

    filepaths = get_all_filepaths()
    for i in range(len(filepaths)):
        filepath = filepaths[i]
        print("Loading from file: " + filepath + " (" + str(i + 1) + "/" + str(len(filepaths)) + ")")
        df = load_df_from_files(filepath, DF_TYPE_TIME_SERIES_DOMAIN_PCA)
        no_features = df.shape[1] - 1
        values = df.iloc[:, 0:no_features]
        labels = df.loc[:, "labelPhone"]
        window_index_start = 0
        window_index_increasing_size = int(WINDOW_SIZE / 2)
        while window_index_start + WINDOW_SIZE < df.shape[0]:
            if random() < test_split:
                test_x.append(values.iloc[i])
                test_y.append(labels[i])
            else:
                train_x.append(values.iloc[i])
                train_y.append(labels[i])
            window_index_start += window_index_increasing_size
        # for i in range(df.shape[0]):
        #     if random() < test_split:
        #         test_x.append(values.iloc[i])
        #         test_y.append(labels[i])
        #     else:
        #         train_x.append(values.iloc[i])
        #         train_y.append(labels[i])
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

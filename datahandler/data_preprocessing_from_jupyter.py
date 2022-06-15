import math
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datahandler.constants import all_features, train_folder, location_labels, test_folder, acc_features, mag_features, \
    v4_mix, v4_walking, v4_standing, v4_standing_simplified, v4_mix_labeled, activity_labels
from datahandler.data_loader import load_data_from_file
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.decomposition import PCA
from random import random

DF_TYPE_RAW = 0
DF_TYPE_RAW_ADDED = 1
DF_TYPE_TIME_SERIES_DOMAIN_NORMALIZED = 2
DF_TYPE_TIME_SERIES_DOMAIN_PCA = 3
WINDOW_SIZE = 40
WINDOW_LENGTH_IN_SECONDS = 2
added_features = ["accMag", "gyroMag", "magMag", "accAng", "gyroAng", "magAng"]
test_split = 0.15
data_folder = v4_mix_labeled
top_10_features = ["stdmagAng", "minmagnetometerZ", "maxmagAng", "maxmagnetometerZ", "minmagAng", "minaccelerometerX",
                   "stdmagnetometerY", "stdgyroAng", "mingyroAng", "stdaccelerometerX"]
top_15_features = ["stdmagAng", "minmagnetometerZ", "maxmagAng", "maxmagnetometerZ", "minmagAng", "minaccelerometerX",
                   "stdmagnetometerY", "stdgyroAng", "mingyroAng", "stdaccelerometerX", "minmagMag", "mingyroscopeY",
                   "meangyroscopeX", "maxmagMag", "meangyroscopeZ"]


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
    contain_activity_labels = 'labelActivity' in df.columns
    feature_count = df.shape[1] - 1
    values = df.iloc[:, 0:feature_count]
    phone_labels = df.loc[:, "labelPhone"].set_axis(range(df.shape[0]))
    if contain_activity_labels:
        activity_labels = df.loc[:, "labelActivity"].set_axis(range(df.shape[0]))
    normalizer = MinMaxScaler()
    normalized_values = normalizer.fit_transform(values)
    normalized_df = pd.DataFrame(
        data=normalized_values,
        columns=values.columns
    )
    normalized_df["labelPhone"] = phone_labels
    if contain_activity_labels:
        normalized_df["labelActivity"] = activity_labels
    return normalized_df


def load_df_from_files(filepath, df_type, selected_features=all_features, drop_activity=True):
    # Step 1: Load from data
    df = load_data_from_file(filepath)
    if drop_activity:
        df = df.drop("labelActivity", axis=1)
    else:
        df['labelActivity'] = df['labelActivity'].apply(lambda x: activity_labels.index(x))
    df['labelPhone'] = df['labelPhone'].apply(lambda x: location_labels.index(x))
    for feature in all_features:
        if feature not in selected_features:
            df = df.drop(feature, axis=1)

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

    if df_type == DF_TYPE_RAW_ADDED:
        return normalize_df(fixed_size_df)

    # Step 4: Convert current features into time-series domain feature
    window_index_start = 0
    window_index_increasing_size = int(WINDOW_SIZE / 4)
    feature_columns = []
    feature_data = []

    domain_types = all_features + added_features
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

    # Step 4.2 - Select top features only
    supported_features = top_10_features + ["labelPhone"]
    features_df = features_df[supported_features]

    if df_type == DF_TYPE_TIME_SERIES_DOMAIN_NORMALIZED:
        return normalize_df(features_df)

    if df_type == DF_TYPE_TIME_SERIES_DOMAIN_PCA:
        return standardize_pca_df(features_df)

    raise Exception("Invalid DF type")


######### FINAL FUNCTIONS TO USE - Should return train/test set
def get_all_filepaths():
    res = []
    for datafile in os.listdir(data_folder):
        if datafile.startswith("."):
            continue
        path = os.path.join(data_folder, datafile)
        if os.path.isfile(path):
            res.append(path)
    return res
    # return [
    #     '/Users/duc.letran/Desktop/FINAL PROJECT/context_transformer/data/v4/mix_labeled/tt1_datacollection.csv',
    #     '/Users/duc.letran/Desktop/FINAL PROJECT/context_transformer/data/v4/mix_labeled/tt2_datacollection.csv',
    #     '/Users/duc.letran/Desktop/FINAL PROJECT/context_transformer/data/v4/mix_labeled/tt3_datacollection.csv',
    #     '/Users/duc.letran/Desktop/FINAL PROJECT/context_transformer/data/v4/mix_labeled/tt4_datacollection.csv',
    #     '/Users/duc.letran/Desktop/FINAL PROJECT/context_transformer/data/v4/mix_labeled/tt5_datacollection.csv'
    # ]


def load_all_raw_multitask_data():
    train_x, train_context_y, train_activity_y, test_x, test_context_y, test_activity_y = [], [], [], [], [], []

    filepaths = get_all_filepaths()

    # filepaths = [
    #     '/Users/duc.letran/Desktop/FINAL PROJECT/context_transformer/data/v4/mix_labeled/tt1_datacollection.csv',
    #     '/Users/duc.letran/Desktop/FINAL PROJECT/context_transformer/data/v4/mix_labeled/tt2_datacollection.csv',
    #     '/Users/duc.letran/Desktop/FINAL PROJECT/context_transformer/data/v4/mix_labeled/tt3_datacollection.csv',
    #     '/Users/duc.letran/Desktop/FINAL PROJECT/context_transformer/data/v4/mix_labeled/tt4_datacollection.csv',
    #     '/Users/duc.letran/Desktop/FINAL PROJECT/context_transformer/data/v4/mix_labeled/tt5_datacollection.csv'
    # ]
    for i in range(len(filepaths)):
        filepath = filepaths[i]
        print("Loading from file: " + filepath + " (" + str(i + 1) + "/" + str(len(filepaths)) + ")")
        df = load_df_from_files(filepath, DF_TYPE_RAW, all_features, drop_activity=False)

        window_index_start = 0
        window_index_increasing_size = int(WINDOW_SIZE / 2)
        no_features = df.shape[1] - 2
        values = df.iloc[:, 0:no_features]
        phone_labels = df.loc[:, "labelPhone"]
        activity_labels = df.loc[:, "labelActivity"]
        while window_index_start + WINDOW_SIZE < df.shape[0]:
            if random() < test_split:
                test_x.append(values[window_index_start:(window_index_start + WINDOW_SIZE)])
                test_context_y.append(phone_labels[window_index_start])
                test_activity_y.append(activity_labels[window_index_start])
            else:
                train_x.append(values[window_index_start:(window_index_start + WINDOW_SIZE)])
                train_context_y.append(phone_labels[window_index_start])
                train_activity_y.append(activity_labels[window_index_start])

            window_index_start += window_index_increasing_size

    return np.array(train_x), np.array(train_context_y), np.array(train_activity_y), np.array(test_x), np.array(
        test_context_y), np.array(test_activity_y)

def load_train_test_data_raw_normalized(added_feature=False, selected_features=all_features):
    train_x, train_y, test_x, test_y = [], [], [], []

    filepaths = get_all_filepaths()
    for i in range(len(filepaths)):
        filepath = filepaths[i]
        print("Loading from file: " + filepath + " (" + str(i + 1) + "/" + str(len(filepaths)) + ")")
        if added_feature:
            df_type = DF_TYPE_RAW_ADDED
        else:
            df_type = DF_TYPE_RAW
        df = load_df_from_files(filepath, df_type, selected_features)
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

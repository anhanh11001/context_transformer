import os

import matplotlib.pyplot as plt
import pandas

from datahandler.constants import *


def load_data_from_file(filepath):
    data = pandas.read_csv(filepath)
    data['date'] = pandas.to_datetime(data['date'], format='%d %b %Y %H:%M:%S:%f %z')
    data = data.set_index('date')
    # print("Information about the panda file: Shape " + str(data.shape) + " and some header fields:")
    # print(data.head())
    # print_line_divider()
    return data


def generate_plot_from_file(filepath, label):
    visualised_attributes = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]
    df = load_data_from_file(filepath)

    # Plotting
    fig, ((acc, gyro), (mag, allplots)) = plt.subplots(2, 2)
    for attribute in [acc_x, acc_y, acc_z]:
        acc.set_title("Accelerometer")
        acc.plot(df[attribute], label=attribute)
        acc.axes.get_xaxis().set_visible(False)

    for attribute in [gyro_x, gyro_y, gyro_z]:
        gyro.set_title("Gyroscope")
        gyro.plot(df[attribute], label=attribute)
        gyro.axes.get_xaxis().set_visible(False)

    for attribute in [mag_x, mag_y, mag_z]:
        mag.set_title("Magnetometer")
        mag.plot(df[attribute], label=attribute)
        mag.axes.get_xaxis().set_visible(False)

    for attribute in visualised_attributes:
        allplots.set_title("Combination of all")
        allplots.plot(df[attribute], label=attribute)
        allplots.axes.get_xaxis().set_visible(False)

    # Labelling
    title = "Changes in value of the IMU sensor data over time of label: " + label
    plt.suptitle(title)

    # Saving
    filename = get_file_name_from_file_path(filepath)
    plots_folder = str(os.getcwd()) + "/plots" + data_version
    label_folder = plots_folder + "/" + label.lower().strip()
    if not os.path.exists(label_folder):
        os.mkdir(label_folder)
    plt.savefig(os.path.join(label_folder, filename + ".png"))

    # Displaying
    # plt.show()

    plt.clf()


def get_file_name_from_file_path(filepath):
    splitted = filepath.split("/")
    filename_with_csv = splitted[len(splitted) - 1]
    return filename_with_csv.split(".")[0]


def generate_train_data_plots():
    for label in os.listdir(train_folder):
        if label.startswith("."):
            continue
        label_dir = os.path.join(train_folder, label)
        print("Generating plots for label " + label + "...")
        if os.path.isdir(label_dir):
            for data_file in os.listdir(label_dir):
                if data_file.startswith("."):
                    continue
                filepath = os.path.join(label_dir, data_file)
                generate_plot_from_file(filepath, label)
    print("Plots generated!! See at folders /datahandler/plots.")

## UNCOMMENT THIS AND RUN TO GENERATE PLOTS FOR TRAINING DATA
# generate_train_data_plots()

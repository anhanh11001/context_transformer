import os

import matplotlib.pyplot as plt
import pandas
from datahandler.constants import *
from datahandler.data_loader import load_data_from_file, get_files_from_folder


def plot_features(
        df,
        title,
        features=all_features,
        fragmented_seconds=None,
        show_plot=True,
        save_plot=False,
        save_location=None
):
    fig, axs = plt.subplots(len(features))
    fig.suptitle(title)
    if fragmented_seconds is None:
        plot_df = df
    else:
        end_index = 0
        while end_index < len(df):
            end_index += 1
            if df.index[end_index] - df.index[0] > pandas.Timedelta(seconds=fragmented_seconds):
                break
        plot_df = df[0:end_index]

    for feature_ind in range(len(features)):
        feature = features[feature_ind]
        axs[feature_ind].plot(plot_df[feature])
        axs[feature_ind].set_title(feature, y=0, loc='right', size=7)

    if save_plot and save_location is not None:
        plt.savefig(os.path.join(save_location, title + ".png"))
    if show_plot:
        plt.show()

    if save_plot and save_location is not None:
        plt.clf()


def prepare_empty_plots_folder():
    if not os.path.exists(plots_folder):
        os.mkdir(plots_folder)
    else:
        for path, subdir, filenames in os.walk(plots_folder):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                os.remove(os.path.join(path, filename))


def generate_plots_from_folder(from_folder):
    from_files = get_files_from_folder(from_folder)
    prepare_empty_plots_folder()
    for file in from_files:
        filename_splited = file.split("/")
        filename = filename_splited[len(filename_splited) - 1]
        print("Generating plot for " + filename + "...")
        df = load_data_from_file(file)
        plot_features(
            df=df,
            title="All features: " + filename,
            features=all_features,
            fragmented_seconds=6,
            show_plot=False,
            save_plot=True,
            save_location=plots_folder
        )
    print("All plots generated. See results at folder: " + plots_folder)

# df = load_data_from_file(test_data_file)
# print(df.shape)
generate_plots_from_folder(train_folder)

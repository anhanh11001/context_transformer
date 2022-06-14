import os

import matplotlib.pyplot as plt
import pandas
from datahandler.constants import test_data_file_step_v3, test_data_file_v3, all_features, plots_folder, train_folder, \
    v4_standing
from datahandler.data_loader import load_data_from_file, get_files_from_folder, load_date_from_steptracking_file
from datetime import datetime, timezone, timedelta

from utils import get_project_root


def plot_features(
        df,
        title,
        features=all_features,
        start_seconds=None,
        end_seconds=None,
        show_plot=True,
        save_plot=False,
        save_location=None,
        step_dates=None
):
    if step_dates is None:
        step_dates = []
    fig, axs = plt.subplots(len(features))
    fig.suptitle(title)

    if end_seconds is None:
        end_index = df.shape[0]
    else:
        end_index = 0
        while end_index < len(df):
            end_index += 1
            if df.index[end_index] - df.index[0] > pandas.Timedelta(seconds=end_seconds):
                break
    if start_seconds is None:
        start_index = 0
    else:
        start_index = df.shape[0]
        while start_index >= 0:
            start_index -= 1
            if df.index[start_index] - df.index[0] < pandas.Timedelta(seconds=start_seconds):
                break
        start_index = start_index + 1
    plot_df = df[start_index:end_index]
    plot_start_date = plot_df.index[0].to_pydatetime()
    plot_end_date = plot_df.index[plot_df.shape[0] - 1].to_pydatetime()
    dates = [d for d in step_dates if plot_start_date <= d <= plot_end_date]
    print("Number of steps found: " + str(len(dates)))
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)  # use POSIX epoch

    for feature_ind in range(len(features)):
        feature = features[feature_ind]
        axs[feature_ind].plot(plot_df[feature])
        axs[feature_ind].set_title(feature, y=0, loc='right', size=7)
        for date in dates:
            axs[feature_ind].axvline(x=date, color='r', linestyle='dotted')

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
            end_seconds=None,
            show_plot=False,
            save_plot=True,
            save_location=plots_folder
        )
    print("All plots generated. See results at folder: " + plots_folder)


# file_name = str(get_project_root()) + "/data/v4/mix/mm1_datacollection.csv"
# data_df = load_data_from_file(file_name)
# dates = load_date_from_steptracking_file(test_data_file_step_v3)
# plot_features(
#     df=data_df,
#     title="All features: " + test_data_file_v3,
#     features=all_features,
#     end_seconds=10,
#     show_plot=True,
#     save_plot=False,
#     # step_dates=dates
# )
# generate_plots_from_folder(v4_standing)

from utils import get_project_root
from datetime import datetime

# DATA CONSTANT
date = 'date'
acc_x = 'accelerometerX'
acc_y = 'accelerometerY'
acc_z = 'accelerometerZ'
gyro_x = 'gyroscopeX'
gyro_y = 'gyroscopeY'
gyro_z = 'gyroscopeZ'
mag_x = 'magnetometerX'
mag_y = 'magnetometerY'
mag_z = 'magnetometerZ'
phone_label = 'labelPhone'
activity_label = 'labelActivity'

# FOLDER CONSTANT
data_version = '/v3'
data_folder = str(get_project_root()) + "/data" + data_version
train_folder = data_folder + "/train"
test_folder = data_folder + "/test"
plots_folder = str(get_project_root()) + "/datahandler/plots" + data_version
training_log_folder = str(get_project_root()) + "/models/logs"
test_data_file = str(
    get_project_root()
) + "/data/v1/train/holdinginhand/holdinginhand_data_0aff7db2-582f-4f08-b5d9-1f4742e0eb37.csv"
tensorboard_dir = "tensorboard/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")



# FEATURE LIST
all_features = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]
acc_features = [acc_x, acc_y, acc_z]
gyro_features = [gyro_x, gyro_y, gyro_z]
mag_features = [mag_x, mag_y, mag_z]
location_labels = ['holdinginhand', 'insidethebag', 'lyingonthedesk', 'insidethepantpocket', 'beingusedinhand']

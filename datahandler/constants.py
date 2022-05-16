from utils import get_project_root

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

data_folder = str(get_project_root()) + "/data"
train_folder = data_folder + "/train"
test_folder = data_folder + "/test"
supported_features = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]
location_labels = ['holdinginhand', 'insidethebag', 'lyingonthedesk', 'insidethepantpocket', 'beingusedinhand']

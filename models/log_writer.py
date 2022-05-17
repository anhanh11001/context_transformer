import os
from datetime import datetime
from datahandler.constants import training_log_folder


class LogWriter:

    def __init__(self, enabled=True):
        self.enabled = enabled
        if not enabled:
            return
        self.base_folder = self.prepare_base_folder()
        self.file = open(os.path.join(self.base_folder, "log.txt"), 'w')
        self.write("LOG ON DATE TIME: " + str(datetime.now()))

    def write(self, str, line_divider=False):
        if not self.enabled:
            return
        if line_divider:
            self.file.write("\n*************************************************\n")
        self.file.write(str + "\n")

    def close(self):
        if not self.enabled:
            return
        self.file.close()

    def prepare_base_folder(self):
        date_folder_name = os.path.join(training_log_folder, str(datetime.date(datetime.now())))
        if not os.path.exists(date_folder_name):
            os.mkdir(date_folder_name)

        id_prefix = "id_"
        id = 0
        for folder in os.listdir(date_folder_name):
            if os.path.isdir(os.path.join(date_folder_name, folder)):
                id = max(id, int(folder.removeprefix(id_prefix)))

        id = id + 1
        folder = os.path.join(date_folder_name, id_prefix + str(id))
        os.mkdir(folder)

        return folder

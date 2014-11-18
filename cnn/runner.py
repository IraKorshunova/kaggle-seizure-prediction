import os, shutil, json
from cnn.train import run_trainer
from cnn.predict import run_predictor
from utils.averager import average_submission_files
from utils.config_name_creator import create_cnn_model_name

if __name__ == '__main__':
    settings_files = sorted(os.listdir('settings_dir'))
    for settings_file in settings_files:
        shutil.copy2('settings_dir/'+settings_file, os.getcwd() + '/SETTINGS.json')
        run_trainer()
        run_predictor()

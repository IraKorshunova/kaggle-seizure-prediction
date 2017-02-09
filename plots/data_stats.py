import sys
import utils
from configuration import config, set_configuration
import pathfinder
import preprocess
import loader

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("Usage: train.py <configuration_name>")

    config_name = sys.argv[1]
    set_configuration(config_name)
    expid = utils.generate_expid(config_name)
    print
    print "Experiment ID: %s" % expid
    print

    subjects = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    for subject in subjects:
        print '***********************', subject, '***************************'
        data_path = preprocess.preprocess_data(pathfinder.RAW_DATA_PATH, subject, config().transformation_params)
        x_test, _, idx2filename = loader.load_test_data(data_path, subject)
        n_test_examples = x_test.shape[0]
        n_timesteps = x_test.shape[3]
        clip2label, clip2time, clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
        usage = 'Public'
        n_clips = 0
        interictal = 0
        for c, u in clip2usage.iteritems():
            if u == usage and subject in c:
                n_clips += 1
                if clip2label[c] == 0:
                    interictal += 1
        # print 'n', n_clips
        print str(interictal) + '/' + str((n_clips - interictal))
        # print 'n preictal', n_clips - interictal

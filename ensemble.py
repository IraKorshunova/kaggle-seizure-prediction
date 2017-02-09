import utils
import pathfinder
import csv
from collections import defaultdict
import numpy as np


def make_ensemble(submission_paths, output_path):
    clip2prediction = defaultdict(list)
    for s in submission_paths:
        with open(s, 'rb') as f:
            reader = csv.reader(f)
            reader.next()
            for row in reader:
                clip = row[0]
                prediction = float(row[1])
                clip2prediction[clip].append(prediction)
    clip2avg_pred = {}
    for k, v in clip2prediction.iteritems():
        clip2avg_pred[k] = np.mean(v)
    print clip2avg_pred

    with open(output_path, 'wb') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['clip', 'preictal'])
        for k in sorted(clip2avg_pred):
            csv_writer.writerow([k, str(clip2avg_pred[k])])
    print 'Saved to', output_path


if __name__ == '__main__':
    ensembles_config = ['8b1m_init2', '8b2m', '8b_std_1m']
    submission_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
    submission_paths = []
    for e in ensembles_config:
        submission_paths.append(utils.find_model_metadata(submission_dir, e))
    out_submission_path = submission_dir + '/' + '-'.join(ensembles_config) + '.csv'
    make_ensemble(submission_paths, out_submission_path)

from pandas import DataFrame
from collections import defaultdict

def load_test_labels(csv_path):
    subject_to_df = defaultdict(list)
    d = DataFrame.from_csv(csv_path, index_col=None)
    for i in d.index:
        clip = d['clip'][i]
        preictal = d['preictal'][i]

        subject_name = '_'.join(clip.split('_', 2)[:2])
        subject_to_df[subject_name].append((clip, preictal))

    for subject_name, subject_data in subject_to_df.iteritems():
        subject_to_df[subject_name] = DataFrame(subject_data, columns=['clip', 'preictal'])
    return subject_to_df
transformation_params = {
    'highcut': 180,
    'lowcut': 0.1,
    'nfreq_bands': 8,
    'win_length_sec': 60,
    'features': 'meanlog',
    'stride_sec': 60,
    'hours': True
}

svm_params = {
    'C': 1e6,
    'kernel': 'rbf',
    'gamma': 0.01,
    'probability': False
}

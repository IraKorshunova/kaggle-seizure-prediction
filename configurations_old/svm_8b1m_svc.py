transformation_params = {
    'highcut': 180,
    'lowcut': 0.1,
    'nfreq_bands': 8,
    'win_length_sec': 60,
    'features': 'meanlog',
    'stride_sec': 60,
}

svm_params = {
    'linear_svc' : True,
    'C': 33,
    'probability': False
}
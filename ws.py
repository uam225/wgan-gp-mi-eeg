'''feature extractor'''

import numpy as np
from mne.decoding import CSP
from data_preprocess import get_data, mnebandFilter

def extract_csp_features(sub_id, test_session, data_path):
    # Load and preprocess the data
    data, labels = get_data(sub_id, test_session, data_path)
    data = mnebandFilter(data, labels, 3, 35)
    print('Data shape: ', data.shape)

    # Initialize CSP and extract features
    csp = CSP(n_components=10, reg=None, log=False, norm_trace=False)
    csp_features = csp.fit_transform(data, labels)


    return csp_features
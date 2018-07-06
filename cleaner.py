'''
Interface for training and performing noise-removal

@author: William Tong (wlt2115@columbia.edu)
'''
import pickle
import numpy as np
import simplejson as json
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from model import SineRegModel
from noiserator import generate_waveform

args = None

# assuming freq == 30
s_rate = 1024
time = 10


def get_hyperparams(json_path: str) -> dict:
    with open(json_path) as json_file:
        params_dict = json.load(json_file)
    
    return params_dict

   
reg_params = get_hyperparams('regression.json')


def main(args):
    # TODO for more rigorous testing purposes, repeat with numerous waveforms
    test_waveform = generate_waveform(phase=0, amp=1, freq=30.4,
                                          time=time, s_rate=s_rate)
    pkl_path = 'estimates.pkl'  # first test_waveform, then final_pred
    
    with tf.Session() as sess:
        regressor = SineRegModel(sess, waveform=test_waveform,
                                 time=time, s_rate=s_rate)
        regressor.train(iterations=20, log_every=1, save_every=20)
        final_pred = regressor.predict()
    
    with open(pkl_path, 'wb') as pkl_file:
        pickle.dump(test_waveform, pkl_file)
        pickle.dump(final_pred, pkl_file)

if __name__ == "__main__":
    main(args)
    
    

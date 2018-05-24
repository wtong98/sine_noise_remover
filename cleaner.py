'''
Interface for training and performing noise-removal

@author: William Tong (wlt2115@columbia.edu)
'''

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from model import SineClassifyModel, SineRegModel
from noiserator import generate_waveform

args = None

def main(args):
    with tf.Session() as sess:
        '''
        classifier = SineClassifyModel(session=sess, save_path=r'save', 
                                       time=1, s_rate=128, 
                                       phase=0, amp=1, freq=(1,3))
        classifier.train(iterations=10, log_every=1, save_every=20)
        '''
        
        test_waveform = generate_waveform(phase=0, amp=1, freq=3,
                                          time=1, s_rate=32)
        regressor = SineRegModel(sess, save_path=r'save', waveform=test_waveform,
                                 freq_interval=(2,4), time=1, s_rate=32)
        regressor.train(iterations=10, log_every=1, save_every=20, restore=False)
        

if __name__ == "__main__":
    main(args)
    
    
    
    
    
    
    
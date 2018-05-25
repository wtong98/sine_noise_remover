'''
Interface for training and performing noise-removal

@author: William Tong (wlt2115@columbia.edu)
'''
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from model import SineClassifyModel, SineRegModel
from noiserator import generate_waveform

args = None

# TODO replace with args implementations!
classify_params = {
    'batch_size': 16,
    
    'branch': {
        'layers': 3,
        'start_size': 16,
        'scale_rate': 2,
        'kernel_sizes': [8, 16, 32],
        'strides': 2
    },
    
    'dense': {
        'layers': 2,
        'units': 512,
        'dropout_rate': 0.4
    },
    
    'adam': {
        'epsilon': 0.1
    }    
}

reg_params = {
    'batch_size': 16,
    
    'branch': {
        'layers': 3,
        'start_size': 16,
        'scale_rate': 2,
        'kernel_sizes': [8, 16, 32],
        'strides': 2
    },
    
    'dense': {
        'layers': 2,
        'units': 512,
        'dropout_rate': 0.4
    },
    
    'adam': {
        'epsilon': 0.1
    }    
}

# assuming freq == 30
s_rate = 1024
time = 1 / 32

def main(args):
    # TODO for more rigorous testing purposes, repeat with numerous waveforms
    test_waveform = generate_waveform(phase=0, amp=1, freq=30.4,
                                          time=time, s_rate=s_rate)
    
    with tf.Session() as sess:        
        classifier = SineClassifyModel(session=sess, save_path=r'save_classify', 
                                       time=time, s_rate=s_rate,
                                       phase=0, amp=1, freq=(15,45))
        classifier.train(iterations=10, log_every=1, save_every=20)
        
        pred_logits, _ = classifier.predict(waveforms=test_waveform)
        guess = np.argmax(pred_logits, axis=1).flatten()[0] + 15    
        
        freq_interval = (guess - 2, guess + 2) # TODO need more scientific way
    
    print(freq_interval)
    with tf.Session() as sess: 
        regressor = SineRegModel(sess, save_path=r'save_reg', waveform=test_waveform,
                                 freq_interval=freq_interval, time=time, s_rate=s_rate)
        regressor.train(iterations=10, log_every=1, save_every=20)
        

if __name__ == "__main__":
    main(args)
    
    
    
    
    
    
    
'''
Implements model for inferences

@author: William Tong (wlt2115@columbia.edu)
'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from noiserator import FakeData

### GLOBALS
data = FakeData(freq=60,
                amp=1,
                phase=0)
time = 30
s_rate = 2048

def train_input_fn(batch_size=None):
    global data
    
    dataset = tf.data.Dataset.from_generator(
        lambda: data.generate(time, s_rate), tf.float32)
    
    size = time * s_rate if batch_size is None else batch_size
    dataset = dataset.repeat().batch(size)
    
    return (dataset.make_one_shot_iterator().get_next(),
            tf.constant((data.frequency, data.amplitude, data.phase)))


def conv_model_fn(features, labels, mode):
    pass





def __test():    
    val = train_input_fn()
    with tf.Session() as sess:
        out = sess.run(val)
        out2 = sess.run(val)
    
    print('the plot thickens')
    plt.plot(out[0])
    plt.plot(out2[0])
    plt.show()
    
if __name__ == "__main__":
    __test()

    
    
    
    
    
    
    
    
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

# TODO wrap configurables in argparse
def _draw_data(batch_size=50, samples=200): # samples may or may not be useless
    global data
    
    dataset = tf.data.Dataset.from_generator(
        lambda: data.generate(sets=samples, time=time, s_rate=s_rate), tf.float32)  
    dataset = dataset.repeat().batch(batch_size)
    
    return dataset.make_one_shot_iterator().get_next()


def _build_model(conv_layers=3, dense_layers=2):
    global time, s_rate
    
    input = tf.placeholder_with_default(input=_draw_data(batch_size=5, 
                                                         samples=15), 
                                        shape=tf.TensorShape([None, time * s_rate]))
    # input_layer = tf.reshape(tensor, shape, name)
    # return loss + optimizer

def __conv_layer(input, current_layer):
    if current_layer == 1:
        input_layer = input
    else:
        input_layer = __conv_layer(input, current_layer - 1)
    
    # define conv + pool, figure out proper scaling

def __dense_layer(input, current_layer):
    pass





def __test():    
    val = _draw_data(batch_size=3, samples=3)
    with tf.Session() as sess:
        out = sess.run(val)
        out2 = sess.run(val)
        print(out)
        print(out2)
    
    print('the plot thickens')
    plt.plot(out[0])
    plt.plot(out2[0])
    plt.show()
    print('done')
    
if __name__ == "__main__":
    _build_model()

    
    
    
    
    
    
    
    
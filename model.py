'''
Implements model for inferences

@author: William Tong (wlt2115@columbia.edu)
'''

import numpy as np
import tensorflow as tf

from noiserator import FakeData

### GLOBALS
data = FakeData(freq=60,
                amp=1,
                phase=0)

def train_input_fn(data, batch_size):
    # tf.data.Dataset.from_generator(generator, output_types, output_shapes)
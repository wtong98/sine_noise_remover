'''
Implements model for inferences

@author: William Tong (wlt2115@columbia.edu)
'''

import numpy as np
from pathlib import Path
import tensorflow as tf

from noiserator import FakeData

### GLOBALS
data = FakeData(freq=60,
                amp=1,
                phase=0)
time = 5
s_rate = 512

is_training = False
batch_size = 5
tf.logging.set_verbosity(tf.logging.INFO)

# TODO wrap configurables in argparse
# TODO implement with OOP principles
# TODO logging + clean tests

def _draw_data(batch_size=50):
    global data
    
    dataset = tf.data.Dataset.from_generator(
        lambda: data.generator(time, s_rate), tf.float32)  
    dataset = dataset.batch(batch_size)
    
    return dataset.make_one_shot_iterator().get_next()

def _build_inputs():
    global time, s_rate
    
    inputs = tf.placeholder_with_default(input=_draw_data(batch_size=batch_size), 
                                        shape=tf.TensorShape([None, time * s_rate, 1]))
    return inputs


def _build_branch_model(inputs, branch_layers=3, dense_layers=2):
    branch = __branch_layer(inputs, branch_layers)
    branch_flat = tf.contrib.layers.flatten(branch)
    dense = __dense_layer(branch_flat, dense_layers)
    params = tf.layers.dense(inputs=dense, units=3)

    return params

def __branch_layer(inputs, 
                 current_layer, 
                 start_size=16,
                 scale_rate=2, 
                 kernel_sizes=[8, 16, 32], 
                 strides=2):
    if current_layer == 1:
        input_layer = inputs
    else:
        input_layer = __branch_layer(inputs, current_layer - 1)
        
    branches = [tf.layers.conv1d(
                    inputs=input_layer,
                    filters=start_size * scale_rate * current_layer,
                    kernel_size=ksize,
                    strides=strides,
                    padding='same',
                    activation=tf.nn.relu) for ksize in kernel_sizes]
    branch_layer = tf.concat(values=branches, axis=-1)
    
    return branch_layer
        
        
def _build_explicit_model(inputs):
    conv1 = tf.layers.conv1d(
        inputs=inputs,
        filters=16,
        kernel_size=32,
        strides=2,
        padding='same',
        activation=tf.nn.relu)
    
    conv2 = tf.layers.conv1d(
        inputs=conv1,
        filters=32,
        kernel_size=16,
        strides=2,
        padding='same',
        activation=tf.nn.relu)
    
    conv3 = tf.layers.conv1d(
        inputs=conv2,
        filters=64,
        kernel_size=8,
        strides=2,
        padding='same',
        activation=tf.nn.relu)
    
    conv_flat = tf.contrib.layers.flatten(conv3)
    
    dense1 = tf.layers.dense(
        inputs=conv_flat,
        units=2048,
        activation=tf.nn.relu)
    
    dropout1 = tf.layers.dropout(
        inputs=dense1,
        rate=0.4,
        training=is_training)
    
    dense2 = tf.layers.dense(
        inputs=dropout1,
        units=2048,
        activation=tf.nn.relu)
    
    dropout2 = tf.layers.dropout(
        inputs=dense2,
        rate=0.4,
        training=is_training)
    
    params = tf.layers.dense(inputs=dropout2, units=3)
    
    return params

def _build_recursive_model(inputs, conv_layers=3, dense_layers=2):    
    conv = __conv_layer(inputs, conv_layers)
    conv_flat = tf.contrib.layers.flatten(conv)
    dense = __dense_layer(conv_flat, dense_layers)
    params = tf.layers.dense(inputs=dense, units=3)

    return params


def _build_loss(inputs, params):
    phase_list = params[:,0]
    amplitude_list = params[:,1]
    frequency_list = params[:,2]
    
    waveform_list = [_waveform(phase_list[i], amplitude_list[i], frequency_list[i])
                        for i in range(batch_size)]
    mse_list = [tf.losses.mean_squared_error(labels=tf.reshape(inputs[i], [-1]), 
                                             predictions=waveform_list[i])
                        for i in range(batch_size)]
    
    loss = tf.reduce_sum(mse_list)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
    return (loss, train_op)
    
    
def _waveform(phase, amplitude, frequency):
    x = tf.lin_space(start=0.0,
                    stop=2 * np.pi * time,
                    num=s_rate * time) + phase
    waveform = amplitude \
                * tf.sin(x * frequency)
    
    return waveform

def __conv_layer(inputs, 
                 current_layer, 
                 start_size=16,
                 scale_rate=2, 
                 kernel_size=5, 
                 pool_size=2):
    if current_layer == 1:
        input_layer = inputs
    else:
        input_layer = __conv_layer(inputs, current_layer - 1)

    
    conv = tf.layers.conv1d(
        inputs=input_layer,
        filters=start_size * scale_rate * current_layer,
        kernel_size=kernel_size,
        padding='same',
        activation=tf.nn.relu)
    
    pool = tf.layers.max_pooling1d(
        inputs=conv,
        pool_size=pool_size,
        strides=pool_size)
    
    return pool
    

def __dense_layer(inputs, current_layer, 
                  num_units=2048,
                  dropout_rate=0.4):
    global is_training
    
    if current_layer == 1:
        input_layer = inputs
    else:
        input_layer = __dense_layer(inputs, current_layer - 1)

    dense = tf.layers.dense(
        inputs=input_layer,
        units=num_units,
        activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=dropout_rate,
        training=is_training)
    
    return dropout


### Putting things together
inputs = _build_inputs()
params = _build_branch_model(inputs, branch_layers=2, dense_layers=2)
loss, train_op = _build_loss(inputs, params)
    
def train(sess, iterations=2000, log_every=50, save_every=100):
    global loss, train_op, is_training
    
    saver = tf.train.Saver()
    ckpt_path = Path(r'save/model.ckpt')
    if ((ckpt_path.parent / 'checkpoint').exists()):
        saver.restore(sess, str(ckpt_path))
        tf.logging.info("Model restored from %s" % ckpt_path)           
    
    for step in range(iterations):
        is_training = True
        sess.run(train_op)
        if step % log_every == 0:
            params = predict(sess).flatten()
            tf.logging.info("Step %d of %d: \n \
                loss = %.4f\n \
                predictions: phase=%.4f amp=%.4f freq=%.4f" 
                    % (step + 1, iterations, 
                       sess.run(loss),
                       params[0], params[1], params[2]))
        if step % save_every == 0 and step != 0:
            tf.logging.info("Saving model to %s" % ckpt_path)
            saver.save(sess, str(ckpt_path))
    
    tf.logging.info("Performing final save to %s" % ckpt_path)
    saver.save(sess, str(ckpt_path))
                

def predict(sess, waveform=None):
    global inputs, params, is_training
    is_training = False
    
    if waveform is not None:
        waveform = np.reshape(waveform, (-1, time * s_rate, 1))
        pred_params = sess.run(params, feed_dict={inputs: waveform})
    else:
        pred_params = sess.run(params)
    
    return pred_params        

    
if __name__ == "__main__":
    test_waveform = data.generate_waveform(time=time, s_rate=s_rate)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(sess, iterations=10, log_every=1, save_every=20)
        pred_params = predict(sess, test_waveform)
    
    print(pred_params)

    
    
    
    
    
    
    
    
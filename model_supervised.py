'''
Implements model_reg for inferences (supervised)

@author: William Tong (wlt2115@columbia.edu)
'''

import numpy as np
from pathlib import Path
import tensorflow as tf

from noiserator import FakeData

### GLOBALS
data = FakeData(freq=(0,6),
                amp=1,
                phase=0)
time = 5
s_rate = 128

is_training = False
batch_size = 8
tf.logging.set_verbosity(tf.logging.INFO)

# TODO wrap configurables in argparse
# TODO implement with OOP principles
# TODO logging + clean tests

def _draw_data(batch_size=32):
    global data
    
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: data.generator_supervised(
            time=time, 
            s_rate=s_rate),
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([time * s_rate, 1]), tf.TensorShape([])))
    dataset = dataset.batch(batch_size)
    
    return dataset.make_one_shot_iterator().get_next()


def _build_inputs():
    global time, s_rate
    train_data, labels = _draw_data(batch_size)
    inputs = tf.placeholder_with_default(input=train_data, 
                                        shape=tf.TensorShape([None, time * s_rate, 1]))
    return (inputs, labels)


def _build_branch_model(inputs, branch_layers=3, dense_layers=2):
    branch = __branch_layer(inputs, branch_layers)
    branch_flat = tf.contrib.layers.flatten(branch)
    dense = __dense_layer(branch_flat, dense_layers)
    logits = tf.layers.dense(inputs=dense, units=6)

    return logits

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
        
        
def __dense_layer(inputs, current_layer, 
                  num_units=512,
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

     
def _build_loss(labels, logits):    
    onehot_labels = tf.one_hot(labels, depth=6)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)
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


### Putting things together
inputs, labels = _build_inputs()
logits = _build_branch_model(inputs, branch_layers=2, dense_layers=2)
loss, train_op = _build_loss(labels, logits)
    
def train(sess, iterations=2000, log_every=50, save_every=100):
    global loss, train_op, labels, is_training
    
    saver = tf.train.Saver()
    ckpt_path = Path(r'save/model_reg.ckpt')
    if ((ckpt_path.parent / 'checkpoint').exists()):
        saver.restore(sess, str(ckpt_path))
        tf.logging.info("Model restored from %s" % ckpt_path)           
    
    for step in range(iterations):
        is_training = True
        sess.run(train_op)
        if step % log_every == 0:
            _evaluate(sess, step, iterations, sess.run(loss)) 
        if step % save_every == 0 and step != 0:
            tf.logging.info("Saving model_reg to %s" % ckpt_path)
            saver.save(sess, str(ckpt_path))
    
    tf.logging.info("Performing final save to %s" % ckpt_path)
    saver.save(sess, str(ckpt_path))
                

def _evaluate(sess, step, iterations, loss):
    pred_logits, labs = predict(sess)
    
    guesses = np.argmax(pred_logits, axis=1)    
    assert(len(guesses) == len(labs))
        
    total = len(guesses)
    correct = [1 for i in range(total) if guesses[i] == labs[i]]    
    accuracy= np.sum(correct) / total
    
    tf.logging.info("Step %d of %d: \n \
                loss = %.4f\n \
                predictions:\t %s\n \
                actual:\t %s\n \
                accuracy: %.4f" 
                    % (step, iterations,
                       loss,
                       guesses,
                       labs,
                       accuracy))
    

def predict(sess, waveforms=None, labels=None):
    global inputs, logits, is_training
    is_training = False
    
    if waveforms is not None:
        waveforms = np.reshape(waveforms, (-1, time * s_rate, 1))
    else:
        waveforms, labels = sess.run(_build_inputs())
        
    pred_logits = sess.run(logits, feed_dict={inputs:waveforms})        
    return (pred_logits, labels)

    
if __name__ == "__main__":
    test_waveform = data.generate_waveform(time=time, s_rate=s_rate, freq=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(sess, iterations=10, log_every=1, save_every=20)
        pred_params = predict(sess, test_waveform)
    
    print(pred_params)

    
    
    
    
    
    
    
    
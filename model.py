'''
Grand implementation of two-pronged classification + regression model

@author: William Tong (wlt2115@columbia.edu)
'''

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf

import noiserator

default_params = {
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

class Model(ABC):
    
    @abstractmethod
    def __init__(self, session, save_path,
                 hyperparams=default_params):
        self.params=hyperparams
        
        self.sess = session
        self.save_path = Path(save_path)
        self.batch_size = self.params['batch_size']        
        
        self.is_training = False
        self.saver = None
    
    @abstractmethod
    def _build_model(self):
        pass
    
    def _branch_layer(self, inputs, current_layer):
        start_size = self.params['branch']['start_size']
        scale_rate = self.params['branch']['scale_rate']
        kernel_sizes = self.params['branch']['kernel_sizes']
        strides = self.params['branch']['strides']
                       
        if current_layer == 1:
            input_layer = inputs
        else:
            input_layer = self._branch_layer(inputs, current_layer - 1)
            
        branches = [tf.layers.conv1d(
                        inputs=input_layer,
                        filters=start_size * scale_rate * current_layer,
                        kernel_size=ksize,
                        strides=strides,
                        padding='same',
                        activation=tf.nn.relu) for ksize in kernel_sizes]
        branch_layer = tf.concat(values=branches, axis=-1)
        
        return branch_layer
    
    def _dense_layer(self, inputs, current_layer):
        num_units = self.params['dense']['units']
        dropout_rate = self.params['dense']['dropout_rate']
    
        if current_layer == 1:
            input_layer = inputs
        else:
            input_layer = self._dense_layer(inputs, current_layer - 1)
    
        dense = tf.layers.dense(
            inputs=input_layer,
            units=num_units,
            activation=tf.nn.relu)
        
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=dropout_rate,
            training=self.is_training)
    
        return dropout
    
    
    def train(self, iterations, log_every=10, save_every=500, restore=True):        
        self._build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        ckpt_path = self.save_path / 'model.ckpt'
        if (restore):
            self._restore(ckpt_path)
    
        for step in range(iterations):
            self.is_training = True
            self.sess.run(self.train_op)
            
            if step % log_every == 0:
                tf.logging.info("Step %d of %d: loss = %.4f" % 
                    (step, iterations, self.sess.run(self.loss)))
                print(self._prettyify(self.eval()))
            if step % save_every == 0 and step != 0:
                self._save(ckpt_path)
        
        self._save(ckpt_path)    
        tf.logging.info('Done!')
    
    def _restore(self, ckpt_path):
        if (ckpt_path.parent.exists()):
            self.saver.restore(self.sess, str(ckpt_path))
            tf.logging.info("Model restored from %s" % ckpt_path)
        else:
            tf.logging.warn("Checkpoints not found at: %s, making new one" 
                            % ckpt_path)
    
    def _save(self, ckpt_path):
        tf.logging.info("Saving model to %s" % ckpt_path)
        self.saver.save(self.sess, str(ckpt_path))
        
    @abstractmethod
    def eval(self) -> dict:
        pass
    
    def _prettyify(self, eval_dict):
        return str(eval_dict)
    
    @abstractmethod
    def predict(self):
        pass


class SineClassifyModel(Model):
    
    def __init__(self, session, save_path,
                 time, s_rate,
                 phase, amp, freq,
                 hyperparams=default_params):
        super(SineClassifyModel, self).__init__(session, save_path, hyperparams)
        
        self.time = time
        self.s_rate = s_rate
        
        self.phase = phase
        self.amp = amp
        self.freq = freq
        
        self.num_classes = freq[1] - freq[0] + 1
        
    def _draw_data(self):
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: noiserator.generator(
                phase = self.phase, amp = self.amp, freq_interval=self.freq,
                time=self.time, s_rate=self.s_rate),
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([self.time * self.s_rate, 1]), 
                           tf.TensorShape([])))
        dataset = dataset.batch(self.batch_size)
        
        return dataset.make_one_shot_iterator().get_next()
    
    def _build_inputs(self):
        train_data, self.labels = self._draw_data()
        self.inputs = tf.placeholder_with_default(
            input=train_data,
            shape=tf.TensorShape([None, self.time * self.s_rate, 1]))
    
    def _build_branch_model(self):
        branch = self._branch_layer(self.inputs, self.params['branch']['layers'])
        branch_flat = tf.contrib.layers.flatten(branch)
        dense = self._dense_layer(branch_flat, self.params['dense']['layers'])
        
        self.logits = tf.layers.dense(inputs=dense, units=self.num_classes)    
    
    def _build_loss(self):
        onehot_labels = tf.one_hot(
            indices=tf.cast(tf.round(self.labels - self.freq[0]), tf.int32),
            depth=self.num_classes)
        
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels, self.logits)
        optimizer = tf.train.AdamOptimizer(self.params['adam']['epsilon'])
        self.train_op = optimizer.minimize(
            loss=self.loss,
            global_step=tf.train.get_global_step())
    
    def _build_model(self):
        self._build_inputs()
        self._build_branch_model()
        self._build_loss()
    
    def eval(self):
        self.is_training = False
        pred_logits, labs = self.predict()
    
        guesses = np.argmax(pred_logits, axis=1) + self.freq[0]
        assert(len(guesses) == len(labs))
            
        total = len(guesses)
        labs_int = np.round(labs)
        correct = [1 for i in range(total) if guesses[i] == labs_int[i]]    
        accuracy= np.sum(correct) / total
        
        return {'predictions': guesses, 'actual': labs, 'accuracy': accuracy}
    
    def _prettyify(self, info_dict):
        info_tuple = (info_dict['predictions'], 
                      info_dict['actual'], 
                      info_dict['accuracy'])
        output ="predictions:\t%s\nactual:\t%s\naccuracy: %.4f\n" % info_tuple

        return output
    
    def predict(self, waveforms=None, labels=None):
        self.is_training = False
    
        if waveforms is None:
            waveforms, labels = self.sess.run(self._draw_data())
            
        pred_logits = self.sess.run(
                        self.logits, 
                        feed_dict={self.inputs: waveforms})        
        return (pred_logits, labels)
    

class SineRegModel(Model):
    
    def __init__(self, session, save_path,
                 waveform, freq_interval,
                 time, s_rate,
                 hyperparams=default_params):
        super(SineRegModel, self).__init__(session, save_path, hyperparams)
                
        self.time = time
        self.s_rate = s_rate
        
        self.waveform = waveform[:self.time * self.s_rate].astype(np.float32)
        self.freq_interval = freq_interval
    
    def _build_inputs(self):
        data = tf.data.Dataset.from_tensors(self.waveform)
        data = data.repeat().batch(1)

        self.inputs = tf.placeholder_with_default(
            input=data.make_one_shot_iterator().get_next(),
            shape=tf.TensorShape([None, self.time * self.s_rate, 1]))
    
    def _build_branch_model(self):
        branch = self._branch_layer(self.inputs, self.params['branch']['layers'])
        branch_flat = tf.contrib.layers.flatten(branch)
        dense = self._dense_layer(branch_flat, self.params['dense']['layers'])
        
        self.estimates = tf.layers.dense(inputs=dense, units=3)
    
    def _build_loss(self):
        phase_est = self.estimates[0,0]
        amp_est = self.estimates[0,1]
        freq_est = self.estimates[0,2]
        
        waveform_est = self._waveform(phase_est, amp_est, freq_est)
        self.loss = tf.losses.mean_squared_error(
                    labels=tf.reshape(self.inputs[0], [-1]),
                    predictions=waveform_est) \
                + self._bound_loss(freq_est, (
                    self.freq_interval[0], 
                    self.freq_interval[1])) \
                + self._bound_loss(amp_est, (0., 99.)) \
                + self._bound_loss(phase_est, (- np.pi, np.pi))
        
        optimizer = tf.train.AdamOptimizer(self.params['adam']['epsilon'])
        self.train_op = optimizer.minimize(
            loss=self.loss, 
            global_step=tf.train.get_global_step())
    
    def _bound_loss(self, val, interval):
        difference = tf.sign(val - interval[0]) + tf.sign(val - interval[1])
        center = (interval[0] + interval[1]) / 2
        bound_loss = difference ** 2 * (val - center) ** 2
        
        return bound_loss    
        
    def _waveform(self, phase, amplitude, frequency):
        x = tf.lin_space(start=0.0,
                        stop=2 * np.pi * self.time,
                        num=self.s_rate * self.time) + phase
        waveform = amplitude \
                    * tf.sin(x * frequency)
        
        return waveform
       
    def _build_model(self):
        self._build_inputs()
        self._build_branch_model()
        self._build_loss()
        
    def eval(self):
        self.is_training = False
        est = self.sess.run(self.estimates).flatten()   
        return {'phase' : est[0], 'amp' : est[1], 'freq' : est[2]}  
        
    def predict(self):
        self.is_training = False
        return self.eval()





        
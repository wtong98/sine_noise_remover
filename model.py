'''
Grand implementation of regression model

@author: William Tong (wlt2115@columbia.edu)
'''
from pathlib import Path

import numpy as np
import tensorflow as tf

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

class SineRegModel:
    
    def __init__(self, 
                 session: tf.Session,
                 time: float,
                 s_rate: int,
                 save_path: str = None,
                 hyperparams: dict = default_params):
        
        self.params=hyperparams
        
        self.sess = session
        self.save_path = Path(save_path) if save_path is not None else None
        self.batch_size = self.params['batch_size']    
        
        self.is_training = False
        self.saver = None
        
        # TODO: allow exploration in alternative frequencies (return list)
        def get_freq(waveform: np.ndarray) -> float:
            w = np.fft.fft(np.squeeze(waveform))
            freqs = np.fft.fftfreq(len(waveform), d=1/s_rate)
            
            return freqs[np.argmax(abs(w))]            
                
        self.time = time
        self.s_rate = s_rate
        
        self.freq = get_freq(self.waveform)
    
#     def _build_inputs(self) -> None:
#         data = tf.data.Dataset.from_tensors(self.waveform)
#         data = data.repeat().batch(1)
# 
#         self.inputs = tf.placeholder_with_default(
#             input=data.make_one_shot_iterator().get_next(),
#             shape=tf.TensorShape([None, int(self.time * self.s_rate), 1]))

    def _build_inputs(self) -> None:
        #data = tf.data.Dataset.from_generator(generator, output_types, output_shapes)
        pass
    
    def _build_branch_model(self) -> None:
        branch = self._branch_layer(self.inputs, 
                                    self.params['branch']['layers'])
        branch_flat = tf.contrib.layers.flatten(branch)
        dense = self._dense_layer(branch_flat, 
                                  self.params['dense']['layers'])
        
        self.estimates = tf.layers.dense(inputs=dense, units=2)
    
    def _build_loss(self) -> None:
        phase_est = self.estimates[0,0]
        amp_est = self.estimates[0,1]
        
        waveform_est = self._waveform(phase_est, amp_est, self.freq)
        self.loss = tf.losses.mean_squared_error(
                        labels=tf.reshape(self.inputs[0], [-1]),
                        predictions=waveform_est) \
                    + self._bound_loss(amp_est, (0., 99.)) \
                    + self._bound_loss(phase_est, (- np.pi, np.pi))
                    # TODO: improve bound-loss calculation

        
        optimizer = tf.train.AdamOptimizer(epsilon=self.params['adam']['epsilon'])
        self.train_op = optimizer.minimize(
            loss=self.loss, 
            global_step=tf.train.get_global_step())
    
    def _bound_loss(self, center: float, interval: tuple) -> float:
        difference = tf.sign(center - interval[0]) + tf.sign(center - interval[1])
        mid = (interval[0] + interval[1]) / 2
        bound_loss = difference ** 2 * (center - mid) ** 2
        
        return bound_loss
        
    def _waveform(self, 
                  phase: float, 
                  amplitude: float, 
                  frequency: float) -> np.ndarray:
        x = tf.lin_space(start=0.0,
                        stop=2 * np.pi * self.time,
                        num=int(self.time * self.s_rate)) + phase
        waveform = amplitude * tf.sin(x * frequency)
        
        return waveform
       
    def _build_model(self) -> None:
        self._build_inputs()
        self._build_branch_model()
        self._build_loss()
    
    def _branch_layer(self,
                      inputs: tf.Tensor, 
                      current_layer: int) -> tf.Tensor:
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
    
    def _dense_layer(self, 
                     inputs: tf.Tensor, 
                     current_layer: int) -> tf.Tensor:
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
    
    def train(self, 
              iterations: int, 
              log_every: int = 10, 
              save_every: int = 500) -> None:        
        self._build_model()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        # TODO: more elegant way to handle None string type
        ckpt_path = None
        if self.save_path is not None:
            ckpt_path = self.save_path / 'model.ckpt'
        self._restore(ckpt_path)
    
        for step in range(iterations):
            self.is_training = True
            self.sess.run(self.train_op)
            
            if step % log_every == 0:
                tf.logging.info("Step %d of %d: loss = %.4f" % 
                    (step, iterations, self.sess.run(self.loss)))
                print(self.eval())
            if self.save_path is not None \
                and step % save_every == 0 \
                and step != 0:
                self._save(ckpt_path)
        
        self._save(ckpt_path)
        tf.logging.info('Done!')
    
    def _restore(self, ckpt_path: Path) -> None:
        if self.save_path is not None:
            if ckpt_path.parent.exists():
                self.saver.restore(self.sess, str(ckpt_path))
                tf.logging.info("Model restored from %s" % ckpt_path)
            else:
                tf.logging.warn("Checkpoints not found at: %s, making new one" 
                                % ckpt_path)
    
    def _save(self, ckpt_path: Path) -> None:
        if self.save_path is not None:
            tf.logging.info("Saving model to %s" % ckpt_path)
            self.saver.save(self.sess, str(ckpt_path))
        
    def eval(self) -> dict:
        self.is_training = False
        est = self.sess.run(self.estimates).flatten()   
        return {'phase' : est[0], 'amp' : est[1], 'freq' : self.freq}  
        
    def predict(self) -> dict:
        self.is_training = False
        return self.eval()





        
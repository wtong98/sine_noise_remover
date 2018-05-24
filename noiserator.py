'''
Generates sine-wave cloaked in white noise

@author: William Tong (wlt2115@columbia.edu)
'''

import numpy as np    
    
def generator(phase, amp, freq_interval, time=1, s_rate=512):
    lower, upper = freq_interval
    
    while True:
        rand_freq = np.random.rand() \
                    * (upper - lower) \
                    + lower
        yield (
            generate_waveform(phase=phase, amp=amp, freq=rand_freq, 
                              time=time, s_rate=s_rate),
            rand_freq)


def generate_waveform(phase, amp, freq,
                      mean=0, sigma=0.25, 
                      time=1, s_rate=512,
                      reshape=True):
    x = np.linspace(start=0, 
                    stop=2 * np.pi * time, 
                    num=s_rate * time,
                    endpoint=True) + phase
    waveform = amp \
                * np.sin(x * freq) \
                + np.random.normal(loc=mean, scale=sigma, size=len(x))
    
    if reshape:
        waveform = np.expand_dims(waveform, -1)
    
    return waveform


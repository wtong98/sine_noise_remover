'''
Generates sine-wave cloaked in white noise

@author: William Tong (wlt2115@columbia.edu)
'''

import numpy as np    
from typing import Iterable
        
def generator(phase: float, 
              amp: float, 
              freq_interval: tuple, 
              time: float = 1, 
              s_rate: int = 512) -> Iterable:
    lower, upper = freq_interval
    
    while True:
        rand_freq = np.random.rand() \
                    * (upper - lower) \
                    + lower
        yield (
            generate_waveform(phase=phase, amp=amp, freq=rand_freq, 
                              time=time, s_rate=s_rate),
            rand_freq)


def generate_waveform(phase: float, 
                      amp: float, 
                      freq: float,
                      mean: float = 0, 
                      sigma: float = 0.25,
                      time: float = 1, 
                      s_rate: int = 512,
                      reshape: bool = True) -> np.ndarray:
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


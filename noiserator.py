'''
Generates sine-wave cloaked in white noise

@author: William Tong (wlt2115@columbia.edu)
'''

import numpy as np
import matplotlib.pyplot as plt

class FakeData:
    
    def __init__(self, freq=60, amp=1, phase=0, mean=0, sigma=0.25):
        self.frequency = freq
        self.amplitude = amp
        self.phase = phase
        self.mean = mean
        self.sigma = sigma
    
    def generator(self, time=30, s_rate=2048):
        while True:
            yield self.generate_waveform(time, s_rate)
    
    def generate_waveform(self, time=30, s_rate=2048):
        x = np.linspace(start=0, 
                        stop=2 * np.pi * time, 
                        num=s_rate * time, 
                        endpoint=True) + self.phase
        waveform = self.amplitude \
                    * np.sin(x * self.frequency) \
                    + np.random.normal(loc=self.mean, scale=self.sigma, size=len(x))
        waveform = np.expand_dims(waveform, axis=-1)
        return waveform


def __test():
    data = FakeData(freq=1, phase = np.pi)
    y = data.generate(time=2, s_rate=512)[0]
    x = np.linspace(0, 2, 512 * 2)
    plt.plot(x,y)
    plt.show()
  
if __name__ == "__main__":  
    __test()

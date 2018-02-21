'''
Generates sine-wave cloaked in white noise

@author: William Tong (wlt2115@columbia.edu)
'''

import numpy as np
import matplotlib.pyplot as plt

class FakeData:
    
    def __init__(self, freq=60, amp=1, phase=0):
        self.frequency = freq
        self.amplitude = amp
        self.phase = phase
        
    def generate(self, time=30, srate=2048):
        rads_per_sec = self.frequency * 2 * np.pi
        for i in (x + self.phase \
                  for x in np.arange(0, rads_per_sec * time, rads_per_sec / srate)):
            yield np.sin(i) + np.random.normal()

def __test():
    data = FakeData(freq=2, phase = np.pi)
    y = [y for y in data.generate(time=1, srate=512)]
    rads_per_sec = data.frequency * 2 * np.pi
    x = [x for x in np.arange(0, rads_per_sec * 1, rads_per_sec / 512)]
    plt.plot(x,y)
    plt.show()
  
if __name__ == "__main__":  
    __test()

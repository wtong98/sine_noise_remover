'''
Simple utilities for visualizing the data

@author: William Tong (wlt2115@columbia.edu)
'''
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import signal

from noiserator import generate_waveform

def load_from_pickle(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        test_waveform = pickle.load(pkl_file)
        final_pred = pickle.load(pkl_file)
        
        return (test_waveform, final_pred)
    
def plot(waveform: np.ndarray, 
         pred_params: dict, 
         s_rate: int, 
         time: float) -> None:
    phase = pred_params['phase']
    amp = pred_params['amp']
    freq = pred_params['freq']
    pred_waveform = generate_waveform(phase, amp, freq, 
                                      mean=0, 
                                      sigma=0, 
                                      time=time, s_rate=s_rate, 
                                      reshape = False)
    x = np.linspace(start=0,
                    stop=time * 2 * np.pi, 
                    num=s_rate * time,
                    endpoint=True) + phase
    plt.plot(x, waveform, '.')
    plt.plot(x, pred_waveform)
    
    plt.ylabel('amplitude')
    plt.xlabel('time (s)')
    plt.title('Data and prediction')


# def spectrogram(waveform: np.ndarray):
#     f, t, Sxx = signal.spectrogram(waveform)
#     print(waveform.shape)
#     print(Sxx)
#     print(Sxx.shape)
#     plt.pcolormesh(t, f, Sxx)
    
    
def main():
    pkl_path = 'estimates.pkl'
    test_waveform, final_pred = load_from_pickle(pkl_path)
    s_rate = 1024
    time = 10
    
    plot(test_waveform, final_pred, s_rate, time)
    # spectrogram(test_waveform)
    plt.show()
        

if __name__ == '__main__':
    main()
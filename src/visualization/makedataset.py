#Python Script to make dataset
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os, sys, json, datetime, random
from scipy.signal import butter, filtfilt

def import_file(file_path):
    '''imports one file
    '''
    try:
        with open(file_path, 'r') as file:
            d = json.load(file)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {file_path}.")
    return d            

def get_signal_data(path):
    '''Taking in the path of the project ex /home/tuckyg/Documents/PHYS-139-GROUP-PROJECT and returns signal data similar to previous hw
    '''
    path = path + '/data/raw/dataSD1500/'
    files = os.listdir(path)
    binsize = 0.025
    y = []
    t=[]
    data_to_save=[]
    for f in files:
        data = import_file(path+f)

#stations contain the signal of the 3 pmts of the wcds and some related parameters
        stations = pd.DataFrame(data['stations'])
        stations.set_index('id', inplace=True)

        
    
        wcd = stations.iloc[0] #select the first station 

        # Assuming y is a list initialized earlier in your code
        y.append(wcd.pmt1)
        time = [np.arange(len(wcd[f'pmt{i + 1}'])) * binsize for i in range(3)]
        t.append(time[0])
        data_to_save.append([time,wcd.pmt1])
    return y, t, data_to_save

def VEM_to_eV(VEM):
    VEM = np.array(VEM)
    '''slope and intercept are claculated using the energy-calibration.ipynb
    '''
    slope = 45.14153715340955
    intercept = 2.3400270892667407
    return ((VEM - intercept) / slope)/10


def generate_noise(length, alpha):
    white_noise = np.random.normal(0, 1, length)
    freq = np.fft.fftfreq(length)
    colored_noise = white_noise / (np.sign(freq)*np.abs(freq)**(alpha / 2) + 1)
    
    return white_noise + colored_noise

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y / 10

def pad_and_truncate(array, pad_length):
    if pad_length < 0:
        raise ValueError("pad_length must be a non-negative integer")
    
    if pad_length > len(array):
        raise ValueError("pad_length cannot be greater than the length of the array")

    # Create a new array with the specified padding and truncated length
    padded_array = np.zeros_like(array)
    padded_array[:pad_length] = 0
    truncated_length = len(array) - pad_length
    padded_array[pad_length:] = array[:truncated_length]

    return padded_array

def SNR(signal):
    signal = np.array(signal)
    clean_signal = signal[np.isfinite(signal)]
    if clean_signal.size == 0:
        return np.nan
    
    max = np.nanmax(clean_signal)
    rms = np.sqrt(np.nanmean(clean_signal ** 2))
    return max/rms if rms != 0 else np.nan



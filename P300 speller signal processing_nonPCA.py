# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:29:45 2020

@author: beniw
"""

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
import scipy.io
import mne


plt.rcParams['figure.figsize'] = 14, 4 

# Importing the dataset

a = np.ones(10)

dataset = scipy.io.loadmat('A02.mat')
#X = dataset.iloc[:, 1:-1].values
#y = dataset.iloc[:, -1].values
data = dataset['data']
channels_temp = (data[0][0][0]).flatten()
channels = []
for i in range(8):
    channels.append(channels_temp[i][0])
X = data[0][0][1]
y = data[0][0][2].flatten()
y_stim = data[0][0][3]
trial = data[0][0][4].flatten()
classes =  data[0][0][5]
classes_stim = data[0][0][6]
gender = data[0][0][7]
age = data[0][0][8]
ALSfrs = data[0][0][9]
onsetALS = data[0][0][10]

plt.plot(y)
#plt.figure()
# plt.plot(X[10000:100100])

sfreq = 256
f_low = 0.1
f_high = 10
# b, a = butter(5,[low,high],btype = 'band', fs = 256)
# X = lfilter(b, a, X)

info = mne.create_info(channels, sfreq, ch_types = ["eeg"]*8)
raw = mne.io.RawArray(X.T, info)
raw.set_montage("standard_1020")

#raw.plot(duration=5, n_channels=8)

#raw.info['bads'] += ['Oz', 'PO7'] 
picks = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 'PO8']

iir_params = dict(order=8, ftype='butter')

raw.filter(f_low, f_high, picks = picks, method = 'iir', iir_params=iir_params, verbose=True)

#raw.notch_filter(np.arange(50, 128, 60), picks=picks, filter_length='auto', phase='zero')

raw.plot(duration=5, n_channels=8)

raw_data = raw.get_data()

event_id = dict(non_target_stim = 1, target_stim = 2)
tmin, tmax = -0.2, 0.8

# stimulus = np.array([])

# for i in range(len(y)):
#     if y[i] != 0:
#         stimulus = np.append(stimulus,i)
        
# stimulus = stimulus.astype(np.int32)


# events = (np.array([stimulus,np.zeros(len(stimulus)),y[stimulus]])).T
events = (np.array([np.arange(0,347704),np.zeros(len(y)),y])).T
events = events.astype(np.int32)

mne.viz.plot_events(events)

#reject_criteria = dict(eeg = 50)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                        picks=picks, baseline=(-0.2, 0),
                        reject=None, preload=True)

#epochs.plot_drop_log()

#epochs['target_stim'].plot_image(picks=['Cz'])

# X = epochs.get_data()
# X_2d = X.reshape(len(X), -1)
# y = epochs.events[:,-1

# from imblearn.over_sampling import RandomOverSampler

# rus = RandomOverSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_resample(X_2d, y)

# X = np.reshape(X_resampled, (len(X_resampled),8,257))

# from mne.decoding import UnsupervisedSpatialFilter
# from sklearn.decomposition import PCA

# pca = UnsupervisedSpatialFilter(PCA(n_components = 6), average=False)
# #pca = PCA(n_components = 6)
# pca_data = pca.fit_transform(X)

X = epochs.get_data()
epochs.save('epoch-epo-A02.fif')


evokeds = []

for trial_type in epochs.event_id:
    evokeds.append(epochs[trial_type].average())

evoked_diff = mne.combine_evoked([evokeds[1], evokeds[0]], weights=[1, -1])

mne.viz.plot_compare_evokeds(dict(target_stim = evokeds[1], non_target_stim = evokeds[0], diff = evoked_diff ),
                             legend='upper left', show_sensors='upper right', picks = picks)




# mne.write_evokeds('evoked-ave.fif', [target_evoked, non_target_evoked, nothing_evoked])
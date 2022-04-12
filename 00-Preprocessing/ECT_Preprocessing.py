import sys
import mne
from mne.preprocessing import ICA, compute_proj_eog
import numpy as np

# Data path

directory = '/Users/sydneysmith/Projects/ECT/NEW_ORGANIZATION/02-Data/RawEDF/MultiSession/'

# Ask for filename
file = input('Please enter raw EDF filename: \n') #'[XXX] (Pre) Session 1 (12.02.19).edf'
path = directory + file
save_path = '/Users/sydneysmith/Projects/ECT/NEW_ORGANIZATION/02-Data/RawFIF/'
save_filename = str(file[:-4])+'raw.fif'   

# For NPZ files
# directory = '/Users/sydneysmith/Projects/ECT/NEW_ORGANIZATION/02-Data/RawNPZ/MutliSession/'
# file = '[XXX] (Pre) Session 12 (12.08.17).npz'
# path = directory + file

# Import MNE raw object
# For EDF
print('\n loading'+str(file)+'\n\n')
raw = mne.io.read_raw_edf(path).load_data()

# get sampling rate
fs = int(raw.info['sfreq'])

# For NPZ
# eeg_dat = np.load(path)
# data = eeg_dat['data']
# ch_names = list(eeg_dat['ch_names'])
# ch_types = list(eeg_dat['ch_types'])
# sfreq = eeg_dat['sfreq']
# info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

# raw = mne.io.RawArray(data, info)

# Specify channels types
ch_names = list(raw.info['ch_names'])

ch_types = ['misc','misc',
           'eeg','eeg','eeg',
           'eeg','eeg','eeg',
           'eeg','eeg','eeg',
           'eeg','eeg','eeg',
           'eeg','eeg',
           'misc','misc','misc',
           'misc','misc','misc',
           'misc','misc','misc',
           'misc','misc','misc',
           'misc','misc','misc',
           'misc','misc','misc',
           'misc','misc','misc',
           'misc','misc','misc']

ch_types = dict(zip(ch_names, ch_types))

raw.set_channel_types(ch_types)
raw.set_montage('standard_1020')

l_freq = 1.0
h_freq = None
raw.filter(l_freq, h_freq)
raw.info['bads']= []

scalings = {'eeg': 20e-5/2}

raw.plot(duration= 15.0, start= 0.0, n_channels= 14, scalings=scalings,
         bad_color= (0.8, 0.8, 0.8));


run_ICA = input('Run ICA? (y/n): \n')
if run_ICA == 'n':
  sys.exit() 


method = 'fastica'

# Choose other parameters
n_components = 14  # if float, select n_components by explained variance of PCA
decim = 3  # we need sufficient statistics, not all time points -> saves time

# we will also set state of the random number generator - ICA is a
# non-deterministic algorithm, but we want to have the same decomposition
# and the same order of components each time this tutorial is run
random_state = 23

print('\nfitting ICA \n')

ica = ICA(n_components=n_components, method=method, random_state=random_state)
print(ica)
ica.fit(raw, picks='eeg', decim=decim)
print(ica)

ICA_detect = input('\nDetect eyeblink with time segment? (y/n): \n')

if ICA_detect=='y':
  start_find=input('eyeblink start time: \n')
  stop_find=input('eyeblink stop time: \n')
  # Detect artifacts
  ica.detect_artifacts(raw, start_find=float(start_find), stop_find=float(stop_find))

plot_ICA = input('\nPlot ICA components? (y/n): \n')
if plot_ICA=='y':
  ica.plot_components()
  ica.plot_properties(raw, picks='eeg')

exclude_components = input('\nPlease enter components to exclude: \n')

print('\nexcluding components \n')

ica.exclude.extend([int(exclude_components)])
raw_copy = raw.copy()
print('\napplying ICA to raw data \n')
ica.apply(raw_copy)

print('\nPlease annotate the data \n   - mark good segments as "GOOD" \n   - click bad channels to mark as bad \n\n')

raw_copy.plot(duration= 15.0, start= 0.0, n_channels= 14, scalings=scalings,
         bad_color= (0.8, 0.8, 0.8))

view_psd = input('\nView PSD? (y/n): \n')

if view_psd == 'y':
  tmin = input('PSD time segment start: \n')
  tmax = input('PSD time segment end: \n')
  raw_copy.plot_psd(fmin=0, fmax=60, tmin=int(tmin), tmax=int(tmax),
                    n_fft=int(fs*2), n_overlap=fs, picks='eeg')


save_dat = input('\nSave file? (y/n): \n')


if save_dat=='n':
  sys.exit()
else:
  raw_copy.save(save_path+save_filename, overwrite=True)
  sys.exit()

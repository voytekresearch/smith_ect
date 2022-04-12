import os
import mne
import numpy as np
import scipy as sp
import pandas as pd
from neurodsp import spectral
from fooof import FOOOF
from fooof import analysis
from fooof.utils import trim_spectrum
import matplotlib.pyplot as plt

# Channel Indicies and Labels
channels = [0,1,2,11,12,13]
labels = {0:'AF3', 1:'F7', 2:'F3', 11:'F4', 12:'F8', 13:'AF4'}

###########################################################################
################################ FUNCTIONS ################################
###########################################################################

def make_empty_lists(ratings_df):
    """Creates empty lists and assigns them to variables in a dictionary form.
    
    Parameters
    ----------
    ratings_df: Pandas DataFrame
        DataFrame containing the clinical ratings of all patients in the study

    Returns 
    -------
    my_list: Dictionary
        Dictionary containing the components within each file as a key
        and the assignment of empty list to each key as the value.
    """

    # get list of column names from ratings dataframe
    ratings_list = ratings_df.columns[3:].to_list()

    from operator import itemgetter
    exp_dict = ['patients', 'sessions', 'preposts', 'chans', 'hemis', 'chans_exps', 'offsets',
                'deltas', 'delta_pows', 'alphas', 'alpha_pows', 'thetas', 'theta_pows', 'delta_band_pow', 
                'alpha_band_pow', 'theta_band_pow', 'l_pow', 'h_pow', 'total_pow']

    # append rating column names to my_list
    for rating in ratings_list:
        exp_dict.append(rating)

    exp_dict = {key:[] for key in exp_dict}
    return exp_dict  

def dir_info(file, signal_path):
    """Establishes the directory of file and identifying specific aspects of the file
    
    Parameters
    ----------
    file: str
        Name of file
    signal_path: str
        Directory name
    
    Returns
    -------
    fif: object
        MNE Raw object from .FIF 
    annot: object
        MNE Annotations object from .FIF
    fs: int 
        Sampling rate
    bads: list
        List of bad channels from .FIF
    """
    
#     signal_path = os.path.dirname('/Users/angel/Downloads/Projects/RawFIF/RawFIF/')

#     for file in os.listdir(signal_path):
#         if file.startswith('.'):
#             pass
#         else:
    fif = mne.io.read_raw_fif(signal_path + '/' + file, verbose=False)
    annot = fif.annotations
    fs = fif.info['sfreq']
    bads = fif.info['bads']

    return fif, annot, fs, bads

def compute_channel(fif, annot, fs, bads, channel):
    """Computes Welch spectrum of one channel
    
    Parameters
    ----------
    fif: object
        MNE raw object read from .fif file
    annot: dict
        MNE dictionary of annotations containing onsets and durations of "GOOD," aritfact-free segments
    bads: list
        List of channels with poor contact or excessive noise
    channel: str
        EEG channel name
    
    Returns
    -------
    spectra: 1d array
        Power spectral density
    freqs: 1d array
        Frequencies at which the measure was calculated.

    """
    
    # concatenate annotations from one channel into single signal
    data = concatenate_annot(fif, annot, channel, fs)

    # compute power spectrum
    output = spectral.compute_spectrum_welch(data, fs, avg_type='median', window='hann',
                                         nperseg=fs*2, noverlap=fs, f_range=(1,30))
    spectra = output[1]
    freqs = output[0]

    return spectra, freqs



def append_lists(exp_dict, patient_vars):
    """Fill the empty list within the dictionary through appending the element to the end of list 
    
    Parameter
    ---------
    exp_dict: dict
        dictionary with keys for each variable
    patient_vars: list
        list of patient variables
    
    Returns
    -------
    exp_dict: dict
        dictionary containing all variables for one channel
    
    """
    for idx, col in enumerate(exp_dict.keys()):
        exp_dict[col].append(patient_vars[idx])

    return exp_dict


def get_FOOOF_params(freqs, spectra):
    """returns variables of parameterized power spectrum using FOOOF (specparam)
    
    Parameters
    ----------
    freqs: 1d array
        frequencies for spectrum analysis
    spectra: 1d array
        spectral power values per frequency (must have same shape as freqs)
    
    Returns
    -------
    delta_cf: float
        central frequecy of largest peak in 1-3 Hz range
    delta_pow: float
        power in largest peak in 1-3 Hz range
    theta_cf: float
        central frequecy of largest peak in 3-8 Hz range
    theta_pow: float
        power in largest peak in 3-8 Hz range
    alpha_cf: float
        central frequecy of largest peak in 8-15 Hz range
    alpha_pow: float
        power in largest peak in 8-15 Hz range
    aper: float
        exponent of the aperiodic component
    offset: float
        offset of the aperiodic component
    r2: float
        model fit r^2
    err: float
        model fit error

    """

    # initialize FOOOF model
    fm = FOOOF(peak_threshold=1.5, peak_width_limits=[1.00,12.0], aperiodic_mode='fixed')

    # fit model
    fm.fit(freqs, spectra, freq_range=[1,30])

    # extract peak parameters
    peak_params = fm.peak_params_

    # Delta
    delta_peak = analysis.periodic.get_band_peak(peak_params, band=[1,3])
    delta_cf = delta_peak[0]
    delta_pow = delta_peak[1]
    
    # Theta
    theta_peak = analysis.periodic.get_band_peak(peak_params, band=[3,8])
    theta_cf = theta_peak[0]
    theta_pow = theta_peak[1]

    # Alpha
    alpha_peak = analysis.periodic.get_band_peak(peak_params, band=[8,15])
    alpha_cf = alpha_peak[0]
    alpha_pow = alpha_peak[1]
    
    # Aperiodic component
    aper = fm.aperiodic_params_[1]
    offset = fm.aperiodic_params_[0]

    # Model fit
    r2 = fm.r_squared_
    err = fm.error_
    
    return delta_cf, delta_pow, theta_cf, theta_pow, alpha_cf, alpha_pow, aper, offset, r2, err

def get_info(file, channel):
    """returns set of variables containing info for recording
    
    Parameters
    ----------
    file: str
        name of file
    channel: int
        index of channel time series in datafile 
    
    Returns
    -------
    patient: str
        3-character patient identifier
    date: str
        xx.xx.xx date of recording
    session: int
        ECT session of recording
    prepost: str
        recording identity - pre or post-ECT treatment
    chan: str 
        channel name (standard 10-20 montage)
    hemi: str
        channel location in 'Left' or 'Right' hemisphere
    """
    
    patient = file[1:4]
    date = file[-16:-8]

    if 'Session 1' in file:
        if file[22]=='2':
            session = 12
        elif file[21]=='2':
            session = 12
        else:
            session = 1
    elif 'Session 2' in file:
        session = 2
    elif 'Session 3' in file:
        session = 3
    elif 'Session 4' in file:
        session = 4
    elif 'Session 5' in file:
        session = 5
    elif 'Session 7' in file:
        session = 7
    elif 'Session 8' in file:
        session = 8
    elif 'Session 9' in file:
        session = 9

    if 'Pre' in file:
        prepost = 'Pre'
    elif 'Post' in file:
        prepost = 'Post'

    chan = labels[channel]

    if chan in ['AF3', 'F7', 'F3']:
        hemi = 'Left'
    elif chan in ['AF4', 'F8', 'F4']:
        hemi = 'Right'

    return patient, date, session, prepost, chan, hemi

def get_ratings(ratings_df, patient, session):
    """gets row of ratings for single patient & session
    Parameters
    ----------
    ratings_df: pandas DataFrame
        dataframe of clinical ratings
    patient: str
        patient ID
    session: int
        session number

    Returns
    -------
    ratings: 1d array
        array of clinical ratings in order from ratings df
    """
    ratings_list = ratings_df.columns[3:].to_list()

    ratings = ratings_df[(ratings_df['Patient ID']==patient) &
                         (ratings_df['Session #']==session)][ratings_list].values
    # ratings = str(ratings)[1:-1] 

    # ratings = ratings[0]
    # ratings = str(ratings[1:-1]) 
    # ratings = ratings.split(",") 


    return ratings


def concatenate_annot(fif, annot, channel, fs):
    """Creates psuedo-time series of concatenated good data segments (demeaned) for sliding window analysis
    
    Parameters
    ----------
    fif: object
        MNE Raw object
    annot: object
        MNE Annotations object from .FIF w/ onset, description of good time segments
    channel: int
        channel # (associated w/ 10-20 channel name)
    fs: float or int
        sampling rate

    
    Returns
    -------
    sig: ndarray
        numpy matrix of concatenated good data segments in shape (n_channels, n_timepoints)

    """

    sig = np.array([])
    orig=fif.first_samp
    
    for event in annot.__iter__():
        start = int((event['onset']*fs)-orig) # fixes MNE raw.crop bug with annotations
        dur = int(event['duration']*fs)
        end = start + dur
        dat = fif.get_data(picks=channel+2, start=start, stop=end)
        dat = dat.flatten()
        dat = sp.signal.detrend(dat)
        sig = np.concatenate([sig, dat])
    
    return sig[0:int(100*fs)] # return first 100 seconds of data


def get_band_pow(freqs, spectra, band=[]):
    """get average power in frequency band in spectrum

    Parameters
    ----------
    freqs : 1d array
        Frequency values
    spectra : 1d array
        Power spectrum power values
    band : list of [float, float]
        Frequency band definition
    
    Returns
    -------
    band_pow : float
        average power in band
    """
    
    trim_freqs, trim_pows = trim_spectrum(freqs, spectra, f_range=band)
    #band_pow = np.trapz(trim_pows, trim_freqs)
    band_pow = np.mean(trim_pows)
    
    return band_pow



def get_avg_spectra_fits(signal_path, exclude_CRA):
    """get average FOOOF fits of power spectra across frontal channels for each patient
    
    Parameters
    ----------
    signal_path : str
        pathname to .fif data
    exclude_CRA : bool
        whether to exclude subject with repeated session
    
    Returns
    -------
    avg_preSpec : ndarray
        average pre-ECT spectra for each patient across frontal electrodes
    avg_postSpec : ndarray
        average post-ECT spectra for each patient across frontal electrodes
        
    """
    
    n_subjects = 9
    n_sessions = 2
    n_possible_sessions = 4
    n_prepost = 2
    n_chan = 6
    n_spectra = 59

    avg_preSpec = np.zeros(shape=[n_subjects*n_possible_sessions,n_spectra])
    avg_postSpec = np.zeros(shape=[n_subjects*n_possible_sessions,n_spectra])

    subj_preSpec = np.zeros(shape=[n_chan,n_spectra])
    subj_postSpec = np.zeros(shape=[n_chan,n_spectra])
    
    fm = FOOOF(peak_threshold=1.5, peak_width_limits=[1.00,12.0], aperiodic_mode='fixed')
    
    channels = [0,1,2,11,12,13]
    labels = {0:'AF3', 1:'F7', 2:'F3', 11:'F4', 12:'F8', 13:'AF4'}
    
    ind_pre = 0
    ind_post = 0

    ind_pre_avg = 0
    ind_post_avg = 0

    for file in os.listdir(signal_path):
        if file.startswith('.'):
            pass
        else:
            fif = mne.io.read_raw_fif(signal_path+'/'+file, verbose=False)
            annot = fif.annotations
            fs = fif.info['sfreq']
            bads = fif.info['bads']
            ind_pre_chan = 0
            ind_post_chan = 0
            for channel in channels:
                if labels[channel] in bads:
                    pass
                else: 
                    data = concatenate_annot(fif, annot, channel, fs)
                    output = spectral.compute_spectrum_welch(data, fs, avg_type='median', window='hann',
                                                             nperseg=fs*2, noverlap=fs, f_range=(1,30))
                    spectra = output[1]
                    freqs = output[0]

                    fm.fit(freqs, spectra, freq_range=[1,30])
                    psd = fm.fooofed_spectrum_

                    patient, date, session, prepost, chan, hemi = get_info(file, channel)

                    if exclude_CRA == True:
                        if patient == 'CRA':
                            pass
                        else:
                            if prepost == 'Pre':
                                subj_preSpec[ind_pre_chan,:] = psd
                                ind_pre_chan +=1
                            elif prepost == 'Post':
                                subj_postSpec[ind_post_chan,:] = psd
                                ind_post_chan +=1
        # Get average psd of file
            if prepost == 'Pre' :
                preSpecArray = subj_preSpec[~np.all(subj_preSpec == 0, axis=1)]
                avg_preSpec[ind_pre_avg,:] = np.mean(preSpecArray, axis=0)
                ind_pre_avg +=1
            elif prepost == 'Post':
                postSpecArray = subj_postSpec[~np.all(subj_postSpec == 0, axis=1)]
                avg_postSpec[ind_post_avg,:] = np.mean(postSpecArray, axis=0)
                ind_post_avg +=1
                
    return avg_preSpec, avg_postSpec
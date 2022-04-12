"""generates a pandas data frame containing all periodic and aperiodic features from ECT patients"""
import os
import pandas as pd
from ECT_Functions import *

# path to data
signal_path = os.path.dirname('/Users/sydneysmith/Projects/ECT/NEW_ORGANIZATION/02-Data/NEW_RawFIF/')
save_path = '/Users/sydneysmith/Projects/ECT/NEW_ORGANIZATION/02-Data/AnalysisFeatures/'
save_name = 'all_features_meanband_NEW.csv'
ratings_path = '/Users/sydneysmith/Projects/ECT/NEW_ORGANIZATION/02-Data/ECT_Ratings_COMPLETE_NEW.xlsx'
# channel indicies and labels
channels = [0,1,2,11,12,13]
labels = {0:'AF3', 1:'F7', 2:'F3', 11:'F4', 12:'F8', 13:'AF4'}

# import ratings dataframe
ratings_df = pd.read_excel(ratings_path)

# make empty dataframe and dictionaries for data
exp_df = pd.DataFrame()
exp_dict = make_empty_lists(ratings_df)
spectra_dict = {key:[] for key in ['patient', 'session', 'prepost', 'chan', 'freqs', 'spectra']}

##########################################################################################
##################################### ECT ANALYSIS #######################################
##########################################################################################


for file in os.listdir(signal_path):
    if file.startswith('.'):
        # pass .DS files    
        pass
    else:
        print('loading... '+str(file[:15])+'...')
        # for one recording get MNE object and accessory data
        fif, annot, sf, bads = dir_info(file, signal_path)
        
        for channel in channels:
            if labels[channel] in bads:
                # pass bad channels
                pass
            else:
                # get power spectra for channel
                spectra, freqs = compute_channel(fif, annot, sf, bads, channel)
                # get napp parameters 
                delta_cf, delta_pow, theta_cf, theta_pow, alpha_cf, alpha_pow, chan_exp, offset, r2, err = get_FOOOF_params(freqs,spectra)
                # get delta band power
                delta_band_pow = get_band_pow(freqs, spectra, band=[1., 3])
                # get theta band power
                theta_band_pow = get_band_pow(freqs, spectra, band=[3., 8.])
                # get alpha band power
                alpha_band_pow = get_band_pow(freqs, spectra, band=[8., 13.])
                l_pow = get_band_pow(freqs, spectra, band=[8., 10.5])
                h_pow = get_band_pow(freqs, spectra, band=[10.5, 13])
                total_pow = get_band_pow(freqs, spectra, band=[1., 30.])
                # get recording info
                patient, date, session, prepost, chan, hemi = get_info(file, channel)
                # get ratings info
                ratings = get_ratings(ratings_df, patient, session).tolist()
                # create list of patient variables
                patient_vars = [patient, session, prepost, chan, hemi, chan_exp, offset, delta_cf, delta_pow,
                                alpha_cf, alpha_pow, theta_cf, theta_pow, delta_band_pow, alpha_band_pow,
                                theta_band_pow, l_pow, h_pow, total_pow]
                # append ratings to patient vars
                for sublist in ratings: 
                    for rating in sublist: 
                        patient_vars.append(rating) 
                
                exp_dict = append_lists(exp_dict, patient_vars)

                # save spectra
                spectra_dict = append_lists(spectra_dict, [patient, session, prepost, chan, freqs, spectra])



# convert dictionary to pandas dataframe
exp_df = pd.DataFrame(exp_dict)

print('saving...')

# save dataframe as CSV
exp_df.to_csv(save_path+save_name)
np.savez('/Users/sydneysmith/Projects/ECT/NEW_ORGANIZATION/02-Data/spectra.npz',
         patient=spectra_dict['patient'],
         session=spectra_dict['session'], 
         prepost=spectra_dict['prepost'],
         chan=spectra_dict['chan'], 
         freqs=spectra_dict['freqs'],
         spectra=spectra_dict['spectra'])

print('all done!')



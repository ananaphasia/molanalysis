keep_behavioral_info = True
area_of_interest = None # None, V1 or PM
INPUT_FOLDER = '../sensorium/notebooks/data/IM_prezipped' # relative to molanalysis root folder (can change data locations)
OUTPUT_FOLDER = 'MEI_generation/data' # relative to molanalysis root folder

# If you want to save a subset of the data, define these manually here. All three variables have to be defined. Else, leave this blank

import os 
# session_list = np.array([['LPE10885', '2023_10_20']])
# session_list = np.array(session_list)
# folders = [os.path.join(INPUT_FOLDER, 'LPE10885')]
# files = [[folder, os.path.join(folder, '2023_10_20')]
#          for folder in folders]

# Imports
# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023, Champalimaud Center
Anastasia Simonoff, 2024, Bernstein Center for Computational Neuroscience Berlin
"""

# Import general libs
import logging
from sklearn.preprocessing import normalize
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm.auto import tqdm
import sys

# Set working directory to root of repo
current_path = os.getcwd()
# Identify if path has 'molanalysis' as a folder in it
if 'molanalysis' in current_path:
    # If so, set the path to the root of the repo
    current_path = current_path.split('molanalysis')[0] + 'molanalysis'
else:
    raise FileNotFoundError(
        f'This needs to be run somewhere from within the molanalysis folder, not {current_path}')
os.chdir(current_path)
sys.path.append(current_path)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create a StreamHandler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the logging level for the handler
# Create a Formatter and attach it to the handler
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s - %(message)s')
console_handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(console_handler)

logger.info('Current working directory: %s', os.getcwd())

# TODO: Fix this so it outputs correctly during figure generation
rmap_logger = logging.getLogger('rastermap')
rmap_logger.setLevel(logging.WARNING)
rmap_logger.addHandler(console_handler)
rmap_logger.propagate = False

# Import personal lib funcs
from loaddata.session_info import load_sessions
from utils.imagelib import load_natural_images
from utils.explorefigs import *
from loaddata.get_data_folder import get_local_drive


# Updated by Anastasia Simonoff for her local computer, etc. This should be updated for your local computer, too.

savedir = os.path.join(get_local_drive(
), 'Users\\asimo\\Documents\\BCCN\\Lab Rotations\\Petreanu Lab\\Figures\\Images' if os.environ['USERDOMAIN'] == 'ULTINTELLIGENCE' else 'OneDrive\\PostDoc\\Figures\\Images\\')
logger.info(f'Saving figures to {savedir}')

# INPUT_FOLDER = "../sensorium/notebooks/data/IM_prezipped"
# Add Add folders two levels deep from INPUT_FOLDER into a list

# test if folders already defined 
try: 
    folders
except NameError:
    # First level
    folders = [os.path.join(INPUT_FOLDER, name) for name in os.listdir(
        INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, name)) and not "merged_data" in name]
    folders = [x.replace("\\", "/") for x in folders]
    # Second level
    files = [[folder, os.path.join(folder, name).replace('\\', '/')] for folder in folders for name in os.listdir(
        folder) if os.path.isdir(os.path.join(folder, name)) and not "merged_data" in name]
    # only get last value after /
    session_list = [[folder.split("/")[-1], name.split("/")[-1]]
                    for folder, name in files]

    # drop ['LPE10919', '2023_11_08'] because the data is not converted yet
    session_list = [x for x in session_list if x != ['LPE10919', '2023_11_08']]
    session_list

# Load one session including raw data: ################################################
# example session with good responses

# Load sessions lazy: (no calciumdata, behaviordata etc.,)
sessions, nSessions = load_sessions(protocol='IM', session_list=np.array(session_list))

# Load proper data and compute average trial responses:
for ises in tqdm(range(nSessions)):    # iterate over sessions
    sessions[ises].load_respmat(calciumversion='deconv', keepraw=True)

    # Save respmat
    np.save(os.path.join(files[ises][1], 'respmat.npy'), sessions[ises].respmat)


# Load all IM sessions including raw data: ################################################
# sessions,nSessions   = filter_sessions(protocols = ['IM'])
# for ises in range(nSessions):    # iterate over sessions
#     sessions[ises].load_respmat(calciumversion='deconv',keepraw=False)

def replace_nan_with_avg(arr):
    arr = arr.copy()  # Copy the array to avoid modifying the original
    nan_indices = np.where(np.isnan(arr))[0]  # Get indices of NaN values

    for i in nan_indices:
        # Handle cases where NaN is at the start or end of the array
        if i == 0:
            arr[i] = arr[i + 1]
        elif i == len(arr) - 1:
            arr[i] = arr[i - 1]
        else:
            # Replace NaN with the average of adjacent values
            arr[i] = np.nanmean([arr[i - 1], arr[i + 1]])

    return arr

import shutil 
# Save behavior data in sensorium format

idx_to_delete = []

for i, (sess, sess_obj) in enumerate(zip(session_list, sessions)):
    folder_base = f'{OUTPUT_FOLDER}/{sess[0]}/{sess[1]}'

    pupil_size = sess_obj.respmat_pupilarea.reshape(-1, 1)
    change_of_pupil_size = sess_obj.respmat_pupilareaderiv.reshape(-1, 1)
    locotomion_speed = sess_obj.respmat_runspeed.reshape(-1, 1)

    # Data
    folder = f'{folder_base}/data/behavior'
    os.makedirs(folder, exist_ok=True)

    if np.isnan(pupil_size).all() or np.isnan(change_of_pupil_size).all() or np.isnan(locotomion_speed).all():
        logger.warning(
            f'All values in behavior data for session {sess[0]}/{sess[1]} are NaN. Session will be removed.')
        # Drop session from list
        # session_list = np.delete(session_list, i, axis=0)
        # sessions = np.delete(sessions, i, axis=0)
        idx_to_delete.append(i)
        # Remove folder
        shutil.rmtree(folder_base)
        try:
            os.rmdir(folder_base)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(f'../sensorium/notebooks/data/IM_prezipped/{sess[0]}/')
        except FileNotFoundError:
            pass
        continue

    behavior = np.hstack((pupil_size, change_of_pupil_size, locotomion_speed))
    if keep_behavioral_info:
        behavior = replace_nan_with_avg(behavior)
        assert not np.isnan(behavior).any(), 'There are still NaN values in the behavior data'
    else:
        behavior = np.random.default_rng().normal(size=behavior.shape)

    for i in tqdm(range(behavior.shape[0])):
        np.save(f'{folder}/{i}.npy', behavior[i, :])

    # Meta
    # There is also a /stimulus_Frame, but I'm not sure what the difference is
    folder = f'{folder_base}/meta/statistics/behavior/all'
    os.makedirs(folder, exist_ok=True)

    # Mean
    mean = np.mean(behavior, axis=0)
    np.save(f'{folder}/mean.npy', mean)

    # Std
    std = np.std(behavior, axis=0)
    np.save(f'{folder}/std.npy', std)

    # Min
    min = np.min(behavior, axis=0)
    np.save(f'{folder}/min.npy', min)

    # Max
    max = np.max(behavior, axis=0)
    np.save(f'{folder}/max.npy', max)

    # Median
    median = np.median(behavior, axis=0)
    np.save(f'{folder}/median.npy', median)

    # There is also a /stimulus_Frame, but I'm not sure what the difference is
    folder = f'{folder_base}/meta/statistics/behavior/stimulus_Frame'
    os.makedirs(folder, exist_ok=True)

    # Mean
    mean = np.mean(behavior, axis=0)
    np.save(f'{folder}/mean.npy', mean)

    # Std
    std = np.std(behavior, axis=0)
    np.save(f'{folder}/std.npy', std)

    # Min
    min = np.min(behavior, axis=0)
    np.save(f'{folder}/min.npy', min)

    # Max
    max = np.max(behavior, axis=0)
    np.save(f'{folder}/max.npy', max)

    # Median
    median = np.median(behavior, axis=0)
    np.save(f'{folder}/median.npy', median)

logger.warning(f'Removing sessions idx = {idx_to_delete} from sessions lists.')
session_list = np.delete(session_list, idx_to_delete, axis=0)
sessions = np.delete(sessions, idx_to_delete, axis=0)

### Load the natural images:
# natimgdata = load_natural_images(onlyright=True)
natimgdata = load_natural_images()
natimgdata = natimgdata[:, natimgdata.shape[1]//2:, :]  # Only take the right half

# Save natimgdata in sensorium format

for sess, sess_obj in zip(session_list, sessions):
    folder_base = f'{OUTPUT_FOLDER}/{sess[0]}/{sess[1]}'

    folder = f'{folder_base}/data/images'
    os.makedirs(folder, exist_ok=True)

    image_idxs = sess_obj.trialdata['ImageNumber'].values

    for i, imgidx in tqdm(enumerate(image_idxs), total=len(image_idxs)):
        file_name = f'{folder}/{i}.npy'
        img = natimgdata[:, :, imgidx]
        img = np.reshape(img, (-1, img.shape[0], img.shape[1]))
        np.save(file_name, img)  # hacky but works

    # Meta
    # There is also a /stimulus_Frame, but I'm not sure what the difference is
    folder = f'{folder_base}/meta/statistics/images/all'
    os.makedirs(folder, exist_ok=True)

    # Mean
    mean = np.mean(natimgdata)
    np.save(f'{folder}/mean.npy', mean)

    # Std
    std = np.std(natimgdata)
    np.save(f'{folder}/std.npy', std)

    # Min
    min = np.min(natimgdata)
    np.save(f'{folder}/min.npy', min)

    # Max
    max = np.max(natimgdata)
    np.save(f'{folder}/max.npy', max)

    # Median
    median = np.median(natimgdata)
    np.save(f'{folder}/median.npy', median)

    # Meta
    # There is also a /stimulus_Frame, but I'm not sure what the difference is
    folder = f'{folder_base}/meta/statistics/images/stimulus_Frame'
    os.makedirs(folder, exist_ok=True)

    # Mean
    mean = np.mean(natimgdata)
    np.save(f'{folder}/mean.npy', mean)

    # Std
    std = np.std(natimgdata)
    np.save(f'{folder}/std.npy', std)

    # Min
    min = np.min(natimgdata)
    np.save(f'{folder}/min.npy', min)

    # Max
    max = np.max(natimgdata)
    np.save(f'{folder}/max.npy', max)

    # Median
    median = np.median(natimgdata)
    np.save(f'{folder}/median.npy', median)

# Save pupil center data in sensorium format

for sess, sess_obj in zip(session_list, sessions):
    folder_base = f'{OUTPUT_FOLDER}/{sess[0]}/{sess[1]}'
    folder = f'{folder_base}/data/pupil_center'
    os.makedirs(folder, exist_ok=True)
    pupil_x = sess_obj.respmat_pupilx.reshape(-1, 1)
    pupil_y = sess_obj.respmat_pupily.reshape(-1, 1)

    pupil_center = np.hstack((pupil_x, pupil_y))

    if keep_behavioral_info:
        pupil_center = replace_nan_with_avg(pupil_center)

        while np.isnan(pupil_center).any():
            pupil_center = replace_nan_with_avg(pupil_center)
    else:
        pupil_center = np.random.default_rng().normal(size=pupil_center.shape)

    for i in tqdm(range(pupil_center.shape[0])):
        np.save(f'{folder}/{i}.npy', pupil_center[i, :])

    # Meta
    # There is also a /stimulus_Frame, but I'm not sure what the difference is
    folder = f'{folder_base}/meta/statistics/pupil_center/all'
    os.makedirs(folder, exist_ok=True)

    # Mean
    mean = np.mean(pupil_center, axis=0)
    np.save(f'{folder}/mean.npy', mean)

    # Std
    std = np.std(pupil_center, axis=0)
    np.save(f'{folder}/std.npy', std)

    # Min
    min = np.min(pupil_center, axis=0)
    np.save(f'{folder}/min.npy', min)

    # Max
    max = np.max(pupil_center, axis=0)
    np.save(f'{folder}/max.npy', max)

    # Median
    median = np.median(pupil_center, axis=0)
    np.save(f'{folder}/median.npy', median)

    # Meta
    # There is also a /stimulus_Frame, but I'm not sure what the difference is
    folder = f'{folder_base}/meta/statistics/pupil_center/stimulus_Frame'
    os.makedirs(folder, exist_ok=True)

    # Mean
    mean = np.mean(pupil_center, axis=0)
    np.save(f'{folder}/mean.npy', mean)

    # Std
    std = np.std(pupil_center, axis=0)
    np.save(f'{folder}/std.npy', std)

    # Min
    min = np.min(pupil_center, axis=0)
    np.save(f'{folder}/min.npy', min)

    # Max
    max = np.max(pupil_center, axis=0)
    np.save(f'{folder}/max.npy', max)

    # Median
    median = np.median(pupil_center, axis=0)
    np.save(f'{folder}/median.npy', median)

# Add neuron data

for sess, sess_obj in zip(session_list, sessions):
    folder_base = f'{OUTPUT_FOLDER}/{sess[0]}/{sess[1]}'
    folder = f'{folder_base}/meta/neurons'
    os.makedirs(folder, exist_ok=True)

    celldata = sess_obj.celldata.copy()

    # layer
    # V1
    celldata.loc[(celldata['roi_name'] == 'V1') & (
        celldata['depth'] < 250), 'layer'] = 'L2/3'
    celldata.loc[(celldata['roi_name'] == 'V1') & (
        celldata['depth'] >= 250) & (celldata['depth'] < 350), 'layer'] = 'L4'
    celldata.loc[(celldata['roi_name'] == 'V1') & (
        celldata['depth'] >= 350), 'layer'] = 'L5/6'

    # PM
    celldata.loc[(celldata['roi_name'] == 'PM') & (
        celldata['depth'] < 250), 'layer'] = 'L2/3'
    celldata.loc[(celldata['roi_name'] == 'PM') & (
        celldata['depth'] >= 250) & (celldata['depth'] < 325), 'layer'] = 'L4'
    celldata.loc[(celldata['roi_name'] == 'PM') & (
        celldata['depth'] >= 325), 'layer'] = 'L5'
    
    celldata = celldata.loc[celldata['roi_name'] == area_of_interest] if area_of_interest is not None else celldata
    
    # Save celldata to obj
    sess_obj.celldata = celldata.copy()

    num_neurons = len(celldata)
    
    # layer
    np.save(f'{folder}/layer.npy',
            celldata['layer'].to_numpy(dtype='<U32'))
    
    # animal ids
    np.save(f'{folder}/animal_ids.npy',
            np.full((num_neurons, ), sess_obj.animal_id, dtype='<U32'))

    # area
    np.save(f'{folder}/area.npy',
            celldata['roi_name'].to_numpy(dtype='<U32'))

    # cell motor coordinates
    np.save(f'{folder}/cell_motor_coordinates.npy',
            celldata[['xloc', 'yloc', 'depth']].to_numpy(dtype=int))

    # scan idx
    np.save(f'{folder}/scan_idx.npy',
            np.full((num_neurons, ), 0))

    # sessions
    np.save(f'{folder}/sessions.npy',
            celldata['session_id'].to_numpy(dtype='<U32'))

    # unit ids
    np.save(f'{folder}/unit_ids.npy',
            celldata['cell_id'].to_numpy(dtype='<U32'))

# Save responses in sensorium format

for sess, sess_obj in zip(session_list, sessions):
    folder_base = f'{OUTPUT_FOLDER}/{sess[0]}/{sess[1]}'
    folder = f'{folder_base}/data/responses'
    os.makedirs(folder, exist_ok=True)

    responses = sess_obj.respmat
    responses = replace_nan_with_avg(responses)

    celldata = sess_obj.celldata.copy()

    responses = responses[celldata.index.values]

    sess_obj.respmat = responses

    for i in tqdm(range(responses.shape[1])):
        np.save(f'{folder}/{i}.npy', responses[:, i])

    # Meta
    # There is also a /stimulus_Frame, but I'm not sure what the difference is
    folder = f'{folder_base}/meta/statistics/responses/all'
    os.makedirs(folder, exist_ok=True)

    # Mean
    mean = np.mean(responses, axis=1)
    np.save(f'{folder}/mean.npy', mean)

    # Std
    std = np.std(responses, axis=1)
    np.save(f'{folder}/std.npy', std)

    # Min
    min = np.min(responses, axis=1)
    np.save(f'{folder}/min.npy', min)

    # Max
    max = np.max(responses, axis=1)
    np.save(f'{folder}/max.npy', max)

    # Median
    median = np.median(responses, axis=1)
    np.save(f'{folder}/median.npy', median)

    # Meta
    # There is also a /stimulus_Frame, but I'm not sure what the difference is
    folder = f'{folder_base}/meta/statistics/responses/stimulus_Frame'
    os.makedirs(folder, exist_ok=True)

    # Mean
    mean = np.mean(responses, axis=1)
    np.save(f'{folder}/mean.npy', mean)

    # Std
    std = np.std(responses, axis=1)
    np.save(f'{folder}/std.npy', std)

    # Min
    min = np.min(responses, axis=1)
    np.save(f'{folder}/min.npy', min)

    # Max
    max = np.max(responses, axis=1)
    np.save(f'{folder}/max.npy', max)

    # Median
    median = np.median(responses, axis=1)
    np.save(f'{folder}/median.npy', median)

def calculate_tiers(num_images):
    # Split into train/test/validate
    idxs = np.arange(num_images)
    np.random.shuffle(idxs)
    train_idxs = idxs[:int(num_images * 0.75)]
    test_idxs = idxs[int(num_images * 0.75):int(num_images * 0.916)]
    validate_idxs = idxs[int(num_images * 0.916):]
    return train_idxs, test_idxs, validate_idxs

# Add trial data


for sess, sess_obj in zip(session_list, sessions):
    folder_base = f'{OUTPUT_FOLDER}/{sess[0]}/{sess[1]}'
    folder = f'{folder_base}/meta/trials'
    os.makedirs(folder, exist_ok=True)

    trial_data = sess_obj.trialdata.copy()
    num_trials = trial_data.shape[0]

    # album
    # ???
    np.save(f'{folder}/album.npy',
            np.full((num_trials,), 'UNK', dtype='<U32'))

    # animal id
    np.save(f'{folder}/animal_id.npy',
            np.full((num_trials,), sess_obj.animal_id, dtype='<U32'))

    # condition hash
    # ???
    np.save(f'{folder}/condition_hash.npy',
            np.full((num_trials,), 'UNK', dtype='<U32'))

    # frame image class
    # ???
    np.save(f'{folder}/frame_image_class.npy',
            np.full((num_trials,), 'UNK', dtype='<U32'))

    # frame image id
    np.save(f'{folder}/frame_image_id.npy', trial_data['ImageNumber'].values)

    # frame last flip
    # ???
    np.save(f'{folder}/frame_last_flip.npy',
            np.full((num_trials,), 0))

    # frame pre blank period
    trial_data['tOnset'] = pd.to_datetime(trial_data['tOnset'], unit='s')
    trial_data['tOffset'] = pd.to_datetime(trial_data['tOffset'], unit='s')
    trial_data['presentationTime'] = trial_data['tOffset'] - \
        trial_data['tOnset']

    # calculate inter image interval
    trial_data['tOffset_prev'] = trial_data['tOffset'].shift(1)
    trial_data['tIntertrial'] = trial_data['tOnset'] - \
        trial_data['tOffset_prev']
    trial_data['tIntertrial'].fillna(pd.Timedelta(seconds=0.5), inplace=True)
    trial_data['tIntertrial'] = trial_data['tIntertrial'].dt.total_seconds()
    np.save(f'{folder}/frame_pre_blank_period.npy',
            trial_data['tIntertrial'].values)

    # frame presentation time
    np.save(f'{folder}/frame_presentation_time.npy',
            trial_data['presentationTime'].dt.total_seconds())

    # frame_trial_ts
    np.save(f'{folder}/frame_trial_ts.npy', trial_data['tOnset'].apply(
        lambda x: f"Timestamp('{x}')").to_numpy(dtype='<U32'))

    # scan_idx
    # ???
    np.save(f'{folder}/scan_idx.npy',
            np.full((num_trials,), 0))

    # session
    np.save(f'{folder}/session.npy',
            np.full((num_trials,), sess_obj.session_id))

    # tiers
    trial_data['ImageCount'] = trial_data['ImageNumber'].map(
        trial_data['ImageNumber'].value_counts())
    trial_data['ImagePresentation'] = trial_data.groupby(
        'ImageNumber').cumcount() + 1

    # Assign tiers to images
    for i, group in trial_data.groupby('ImageCount'):
        for j, group2 in group.groupby('ImagePresentation'):
            train_idxs, test_idxs, validate_idxs = calculate_tiers(
                group2.shape[0])

            # Update indices
            train_idxs = group2.index[train_idxs]
            test_idxs = group2.index[test_idxs]
            validate_idxs = group2.index[validate_idxs]

            # assign to tiers
            trial_data.loc[train_idxs, 'tiers'] = 'train'
            trial_data.loc[test_idxs, 'tiers'] = 'test'
            trial_data.loc[validate_idxs, 'tiers'] = 'validation'

    np.save(f'{folder}/tiers.npy',
            trial_data['tiers'].to_numpy(dtype='<U32'))

    # trial_idx
    np.save(f'{folder}/trial_idx.npy', trial_data['TrialNumber'].values)
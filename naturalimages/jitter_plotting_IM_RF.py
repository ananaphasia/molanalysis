# -*- coding: utf-8 -*-
"""
This script analyzes neural and behavioral data in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are natural images.
Matthijs Oude Lohuis, 2023, Champalimaud Center
""" 

#%% Imports
# Import general libs
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

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

from sensorium_utility_training_read_config import read_config

run_config = read_config('../Petreanu_MEI_generation/run_config.yaml') # Must be set

RUN_NAME = run_config['RUN_NAME'] # MUST be set. Creates a subfolder in the runs folder with this name, containing data, saved models, etc. IMPORTANT: all values in this folder WILL be deleted.

keep_behavioral_info = run_config['data']['keep_behavioral_info']
area_of_interest = run_config['data']['area_of_interest']
sessions_to_keep = run_config['data']['sessions_to_keep']
OUTPUT_NAME = run_config['data']['OUTPUT_NAME']
INPUT_FOLDER = run_config['data']['INPUT_FOLDER']
OUTPUT_FOLDER = f'../molanalysis/MEI_generation/data/{OUTPUT_NAME}' # relative to molanalysis root folder
OUT_NAME = f'runs/{RUN_NAME}'

# os.chdir('../')  # set working directory to the root of the git repo

# Import personal lib funcs
from loaddata.session_info import load_sessions
from utils.plotting_style import *  # get all the fixed color schemes
from utils.imagelib import load_natural_images
from loaddata.get_data_folder import get_local_drive
from utils.pair_lib import compute_pairwise_anatomical_distance
from utils.rf_lib import *

savedir = os.path.join(f'../Petreanu_MEI_generation/runs/{RUN_NAME}/Plots/RF_analysis')
os.makedirs(savedir, exist_ok=True)

config_file = f'../Petreanu_MEI_generation/runs/{RUN_NAME}/config_m4_ens0/config.yaml'
config = read_config(config_file)
config['model_config']['data_path'] = f'../Petreanu_MEI_generation/runs/{RUN_NAME}/data'
max_jitter = config['model_config']['max_jitter']
print(config)

# %% Load IM session with receptive field mapping ################################################

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
    print(session_list)

if sessions_to_keep != 'all':
    session_list = [x for x in session_list if x in sessions_to_keep]


session_list = np.array([['LPE10885', '2023_10_20']])
# Load sessions lazy: (no calciumdata, behaviordata etc.,)
sessions, nSessions = load_sessions(protocol='IM', session_list=session_list, data_folder = INPUT_FOLDER)

#%% 
for ises in range(nSessions):    # Load proper data and compute average trial responses:
    sessions[ises].load_respmat(calciumversion='deconv', keepraw=False)

#%% Interpolation of receptive fields:
sessions = compute_pairwise_anatomical_distance(sessions)
sessions = smooth_rf(sessions,radius=75,rf_type='Fneu')
sessions = exclude_outlier_rf(sessions) 
sessions = replace_smooth_with_Fsig(sessions) 

#%% Load the output of digital twin model:
if area_of_interest == 'V1':
    statsfile_V1        = f'../Petreanu_MEI_generation/runs/{RUN_NAME}/results/neuron_stats.csv'
    statsdata_V1        = pd.read_csv(statsfile_V1)	
elif area_of_interest == 'PM':
    statsfile_PM        = f'../Petreanu_MEI_generation/runs/{RUN_NAME}/results/neuron_stats.csv'
    statsdata_PM        = pd.read_csv(statsfile_PM)

try:
    statsdata_V1        = pd.read_csv(statsfile_V1)	
except:
    statsdata_V1        = pd.DataFrame(columns=['cell_id', 'jitter', 'jitter_0', 'jitter_1', 'jitter_2', 'jitter_3', 'jitter_4'])

try:
    statsdata_PM        = pd.read_csv(statsfile_PM)
except:
    statsdata_PM        = pd.DataFrame(columns=['cell_id', 'jitter', 'jitter_0', 'jitter_1', 'jitter_2', 'jitter_3', 'jitter_4'])

ises                = 0
statsdata           = pd.concat([statsdata_V1,statsdata_PM]).reset_index(drop=True)

def replace_str(x):
    return x.replace('   ',' ').replace('  ',' ').replace('[ ','').replace(' ]','').replace('[','').replace(']','').split(' ')

jitter_columns = ['jitter', 'jitter_0', 'jitter_1', 'jitter_2', 'jitter_3', 'jitter_4']
g = statsdata[jitter_columns].applymap(lambda x: replace_str(x))

mergedata = pd.DataFrame(np.array(g['jitter'].values.tolist(), dtype=float), columns=['jitter_az_Ftwin', 'jitter_el_Ftwin',])
for i in range(5):
    temp_df = pd.DataFrame(np.array(g[f'jitter_{i}'].values.tolist(), dtype=float), columns=[f'jitter_az_Ftwin_{i}', f'jitter_el_Ftwin_{i}'])
    mergedata = pd.concat([mergedata, temp_df], axis=1)

mergedata['cell_id'] = statsdata['cell_id']
sessions[ises].celldata = sessions[ises].celldata.merge(mergedata, on='cell_id')
# sessions[ises].celldata['rf_r2_Ftwin'] = 0
# sessions[ises].celldata['jitter_az_Ftwin'] = (sessions[ises].celldata['jitter_az_Ftwin']+0.5)*135
# sessions[ises].celldata['jitter_el_Ftwin'] = (sessions[ises].celldata['jitter_el_Ftwin']+0.5)*62 - 53
# for i in range(5):
#     sessions[ises].celldata[f'jitter_az_Ftwin_{i}'] = (sessions[ises].celldata[f'jitter_az_Ftwin_{i}'] + 0.5) * 135
#     sessions[ises].celldata[f'jitter_el_Ftwin_{i}'] = (sessions[ises].celldata[f'jitter_el_Ftwin_{i}'] + 0.5) * 62 - 53
# sessions[ises].celldata['jitter_el_Ftwin'] = (sessions[ises].celldata['jitter_el_Ftwin']+0.5)*62 - 16.7

# #%% Load the output of digital twin model:
# statsfile       = 'E:\\Procdata\\IM\\LPE10885\\2023_10_20\\LPE10885_2023_10_20_neuron_stats.csv'
# statsdata       = pd.read_csv(statsfile)

# g               = statsdata['jitter'].apply(lambda x: x.replace('[ ','').replace(' ]','').replace('   ',' ').replace('  ',' ').replace('[','').replace(']','').split(' '))
# g               = np.array(list(g), dtype=float)

# mergedata       = pd.DataFrame(data=g,columns=['jitter_az_Ftwin','jitter_el_Ftwin'])
# mergedata['cell_id'] = statsdata['cell_id']
# sessions[ises].celldata = sessions[ises].celldata.merge(mergedata, on='cell_id')
# sessions[ises].celldata['rf_r2_Ftwin'] = 0
# sessions[ises].celldata['jitter_az_Ftwin'] = (sessions[ises].celldata['jitter_az_Ftwin']+0.5)*135
# sessions[ises].celldata['jitter_el_Ftwin'] = (sessions[ises].celldata['jitter_el_Ftwin']+0.5)*62 - 53

#%% Make a histogram of jitters, one for each model:
areas       = ['V1', 'PM']
spat_dims   = ['az', 'el']
clrs_areas  = get_clr_areas(areas)
# sig_thr     = 0.001
# sig_thr     = 0.05
# sig_thr     = 0.001
r2_thr      = 0.5
rf_type      = 'F'
# rf_type      = 'Fneu'
rf_type_twin = 'Ftwin'

fig,axes     = plt.subplots(2,2,figsize=(6,6))
for iarea,area in enumerate(areas):
    for ispat_dim,spat_dim in enumerate(spat_dims):
        idx         = (sessions[0].celldata['roi_name'] == area) & (sessions[0].celldata['rf_r2_' + rf_type] < r2_thr)
        # x = sessions[0].celldata[f'jitter_{spat_dim}_{rf_type}'][idx]
        y = sessions[0].celldata[f'jitter_{spat_dim}_{rf_type_twin}'][idx]

        # sns.scatterplot(ax=axes[iarea,ispat_dim],x=x,y=y,s=7,c=clrs_areas[iarea],alpha=0.5)
        sns.histplot(ax=axes[iarea,ispat_dim],data=y, bins=30)
        axes[iarea,ispat_dim].set_title(f'{area} {spat_dim}',fontsize=12)
        axes[iarea,ispat_dim].set_xlabel('Sparse Noise (deg)',fontsize=9)
        axes[iarea,ispat_dim].set_ylabel(f'Dig. Twin Model',fontsize=9)
        # if spat_dim == 'az':
        #     axes[iarea,ispat_dim].set_xlim([-50,135])
        #     axes[iarea,ispat_dim].set_ylim([-50,135])
        #     # axes[iarea,ispat_dim].set_ylim([-0.5,0.5])
        # elif spat_dim == 'el':
        #     axes[iarea,ispat_dim].set_xlim([-150.2,150.2])
        #     axes[iarea,ispat_dim].set_ylim([-150.2,150.2])
            # axes[iarea,ispat_dim].set_ylim([-0.5,0.5])
        # idx = (~np.isnan(x)) & (~np.isnan(y))
        # x =  x[idx]
        idx = ~np.isnan(y)
        y =  y[idx]
        # print(f'x min: {min(x) if len(x) > 0 else "None"}')
        # print(f'x max: {max(x) if len(x) > 0 else "None"}')
        # print(f'y min: {min(y) if len(y) > 0 else "None"}')
        # print(f'y max: {max(y) if len(y) > 0 else "None"}')
        # if len(x) > 0:
        #     axes[iarea,ispat_dim].set_xlim([int(min(x) - 10), int(max(x) + 10)])
        if len(y) > 0:
            axes[iarea,ispat_dim].set_xlim([-max_jitter - 0.1 * abs(max_jitter), max_jitter + 0.1 * abs(max_jitter)])
        # axes[iarea,ispat_dim].text(x=0,y=0.1,s='r = ' + str(np.round(np.corrcoef(x,y)[0,1],3),))
        # if len(x) > 0 and len(y) > 0:
        #     axes[iarea,ispat_dim].text(x=int(min(x) - 5),y=int(min(y) - 5),s='r = ' + str(np.round(np.corrcoef(x,y)[0,1],3),))
plt.tight_layout()
fig.savefig(os.path.join(savedir, f'Jitter_Histogram_{rf_type}_{sessions[0].sessiondata["session_id"][0]}.png'), format='png')


for i in range(5):
    fig,axes     = plt.subplots(2,2,figsize=(6,6))
    for iarea,area in enumerate(areas):
        for ispat_dim,spat_dim in enumerate(spat_dims):
            idx         = (sessions[0].celldata['roi_name'] == area) & (sessions[0].celldata['rf_r2_' + rf_type] < r2_thr)
            # x = sessions[0].celldata[f'jitter_{spat_dim}_{rf_type}'][idx]
            y = sessions[0].celldata[f'jitter_{spat_dim}_{rf_type_twin}_{i}'][idx]

            # sns.scatterplot(ax=axes[iarea,ispat_dim],x=x,y=y,s=7,c=clrs_areas[iarea],alpha=0.5)
            sns.histplot(ax=axes[iarea,ispat_dim],data=y, bins=30)
            axes[iarea,ispat_dim].set_title(f'{area} {spat_dim} Model {i}',fontsize=12)
            axes[iarea,ispat_dim].set_xlabel('Sparse Noise (deg)',fontsize=9)
            axes[iarea,ispat_dim].set_ylabel(f'Dig. Twin Model {i}',fontsize=9)
            # if spat_dim == 'az':
            #     axes[iarea,ispat_dim].set_xlim([-50,135])
            #     axes[iarea,ispat_dim].set_ylim([-50,135])
            #     # axes[iarea,ispat_dim].set_ylim([-0.5,0.5])
            # elif spat_dim == 'el':
            #     axes[iarea,ispat_dim].set_xlim([-150.2,150.2])
            #     axes[iarea,ispat_dim].set_ylim([-150.2,150.2])
            #     # axes[iarea,ispat_dim].set_ylim([-0.5,0.5])
            # idx = (~np.isnan(x)) & (~np.isnan(y))
            # x =  x[idx]
            idx = ~np.isnan(y)
            y =  y[idx]
            # print(f'x min: {min(x) if len(x) > 0 else "None"}')
            # print(f'x max: {max(x) if len(x) > 0 else "None"}')
            # print(f'y min: {min(y) if len(y) > 0 else "None"}')
            # print(f'y max: {max(y) if len(y) > 0 else "None"}')
            # if len(x) > 0:
            #     axes[iarea,ispat_dim].set_xlim([int(min(x) - 10), int(max(x) + 10)])
            if len(y) > 0:
                axes[iarea,ispat_dim].set_xlim([-max_jitter - 0.1 * abs(max_jitter), max_jitter + 0.1 * abs(max_jitter)])
            # axes[iarea,ispat_dim].text(x=0,y=0.1,s='r = ' + str(np.round(np.corrcoef(x,y)[0,1],3),))
            # if len(x) > 0 and len(y) > 0:
            #     axes[iarea,ispat_dim].text(x=int(min(x) - 5),y=int(min(y) - 5),s='r = ' + str(np.round(np.corrcoef(x,y)[0,1],3),))
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f'Jitter_Histogram_{rf_type}_{sessions[0].sessiondata["session_id"][0]}_model_{i}.png'), format='png')
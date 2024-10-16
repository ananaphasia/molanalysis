# -*- coding: utf-8 -*-
"""
Set of function used for analysis of mouse behavior in visual navigation task
Author: Matthijs Oude Lohuis, Champalimaud Research
2022-2025
"""

import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from utils.corr_lib import compute_pairwise_metrics

def plot_rf_plane(celldata,sig_thr=1,rf_type='Fneu'):
    
    areas           = np.sort(celldata['roi_name'].unique())[::-1]
    # vars            = ['rf_azimuth','rf_elevation']
    vars            = ['rf_az_' + rf_type,'rf_el_' + rf_type]

    fig,axes        = plt.subplots(2,len(areas),figsize=(5*len(areas),10))

    for i in range(len(vars)): #for azimuth and elevation
        for j in range(len(areas)): #for areas
            
            idx_area    = celldata['roi_name']==areas[j]
            idx_sig     = celldata['rf_p_Fneu'] < sig_thr
            idx         = np.logical_and(idx_area,idx_sig)
            
            # sns.scatterplot(data = celldata[idx],x='xloc',y='yloc',
            #                 hue=vars[i],ax=axes[i,j],palette='gist_rainbow',size=9,edgecolor="none")
            
            if vars[i]=='rf_az_' + rf_type:
                sns.scatterplot(data = celldata[idx],x='yloc',y='xloc',hue_norm=(-135,135),
                            hue=vars[i],ax=axes[i,j],palette='gist_rainbow',size=9,edgecolor="none")
            elif vars[i]=='rf_el_' + rf_type:
                sns.scatterplot(data = celldata[idx],x='yloc',y='xloc',hue_norm=(-16.7,50.2),
                            hue=vars[i],ax=axes[i,j],palette='gist_rainbow',size=9,edgecolor="none")

            box = axes[i,j].get_position()
            axes[i,j].set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])  # Shrink current axis's height by 10% on the bottom
            axes[i,j].set_xlabel('')
            axes[i,j].set_ylabel('')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            axes[i,j].set_xlim([0,512])
            axes[i,j].set_ylim([0,512])
            axes[i,j].set_title(areas[j] + ' - ' + vars[i],fontsize=15)
            axes[i,j].set_facecolor("black")
            axes[i,j].get_legend().remove()
            axes[i,j].invert_yaxis()

            if vars[i]=='rf_az_' + rf_type:
                norm = plt.Normalize(-135,135)
            elif vars[i]=='rf_el_' + rf_type:
                norm = plt.Normalize(-16.7,50.2)
            sm = plt.cm.ScalarMappable(cmap="gist_rainbow", norm=norm)
            sm.set_array([])
            # Remove the legend and add a colorbar (optional)
            axes[i,j].figure.colorbar(sm,ax=axes[i,j],pad=0.02,label=vars[i])
    plt.suptitle(celldata['session_id'][0])
    plt.tight_layout()

    return fig


def plot_rf_screen(celldata,sig_thr=1,rf_type='Fneu'):
    
    areas           = np.sort(celldata['roi_name'].unique())[::-1]

    fig,axes        = plt.subplots(1,len(areas),figsize=(6*len(areas),3))

    for j in range(len(areas)): #for areas
        idx_area    = celldata['roi_name']==areas[j]
        idx_sig     = celldata['rf_p_' + rf_type] < sig_thr
        idx         = np.logical_and(idx_area,idx_sig)

        sns.scatterplot(data = celldata[idx],
                        x='rf_az_' + rf_type,y='rf_el_' + rf_type,
                        hue='rf_p_' + rf_type,ax=axes[j],palette='hot_r',size=9,edgecolor="none")

        box = axes[j].get_position()
        axes[j].set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])  # Shrink current axis's height by 10% on the bottom
        # axes[i,j].legend(loc='center left', bbox_to_anchor=(1, 0.5))        # Put a legend next to current axis
        axes[j].set_xlabel('')
        axes[j].set_ylabel('')
        axes[j].set_xticks([])
        axes[j].set_yticks([])
        axes[j].set_xlim([-135,135])
        axes[j].set_ylim([-16.7,50.2])
        axes[j].set_title(areas[j],fontsize=15)
        axes[j].set_facecolor("black")
        axes[j].get_legend().remove()
        # if j==1:
        norm = plt.Normalize(celldata['rf_p_' + rf_type][idx].min(), celldata['rf_p_' + rf_type][idx].max())
        # norm = plt.Normalize(celldata['rf_p_Fneu'][idx].max(), celldata['rf_p_Fneu'][idx].min())
        sm = plt.cm.ScalarMappable(cmap="hot_r", norm=norm)
        sm.set_array([])
        # Remove the legend and add a colorbar (optional)
        axes[j].figure.colorbar(sm,ax=axes[j],pad=0.02,label='rf_p_' + rf_type)
    plt.suptitle(celldata['session_id'][0])
    plt.tight_layout()

    return fig


def interp_rf(sessions,sig_thr=0.001,show_fit=False):

    for ses in sessions:

        if show_fit:
            plot_rf_plane(ses.celldata,sig_thr=sig_thr)

        areas           = np.sort(ses.celldata['roi_name'].unique())[::-1]
        vars            = ['rf_azimuth','rf_elevation']

        # if show_fit:
        #     fig,axes        = plt.subplots(2,2,figsize=(5*len(areas),10))

        r2 = np.empty((2,2))

        for i in range(len(vars)): #for azimuth and elevation
            for j in range(len(areas)): #for areas
                
                idx_area    = ses.celldata['roi_name']==areas[j]
                idx_sig     = ses.celldata['rf_p'] < sig_thr
                idx         = np.logical_and(idx_area,idx_sig)  

                areadf      = ses.celldata[idx].dropna()
                X           = np.array([areadf['xloc'],areadf['yloc']])
                y           = np.array(areadf[vars[i]])

                reg         = LinearRegression().fit(X.T, y)

                r2[i,j]     = r2_score(y,reg.predict(X.T))

                idx         = np.logical_and(idx_area,~idx_sig)  
                ses.celldata.loc[ses.celldata[idx].index,vars[i]] = reg.predict(ses.celldata.loc[ses.celldata[idx].index,['xloc','yloc']].to_numpy())

                # if show_fit:
                #      sns.scatterplot(data = celldata[idx],x='xloc',y='yloc',
                #             hue=vars[i],ax=axes[i,j],palette='gist_rainbow',size=9,edgecolor="none")
            
                #     box = axes[i,j].get_position()
                #     axes[i,j].set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])  # Shrink current axis's height by 10% on the bottom
                #     axes[i,j].set_xlabel('')
                #     axes[i,j].set_ylabel('')
                #     axes[i,j].set_xticks([])
                #     axes[i,j].set_yticks([])
                #     axes[i,j].set_xlim([0,512])
                #     axes[i,j].set_ylim([0,512])
                #     axes[i,j].set_title(areas[j] + ' - ' + vars[i],fontsize=15)
                #     axes[i,j].set_facecolor("black")
                #     axes[i,j].get_legend().remove()

        if show_fit:
            plot_rf_plane(ses.celldata,sig_thr=1)
    
    return r2

def smooth_rf(sessions,sig_thr=0.001,radius=50):

    for ses in sessions:

        idx_RF = ses.celldata['rf_p'] < sig_thr
        idx_RF = np.logical_and(ses.celldata['rf_azimuth']>0,idx_RF)

        for iN in np.where(~idx_RF)[0]:
            idx_near = np.logical_and(np.abs(ses.distmat_xy[iN,:]) < radius,idx_RF)
            ses.celldata.loc[iN,'rf_azimuth']       = np.nanmedian(ses.celldata.loc[ses.celldata[idx_near].index,'rf_azimuth'])
            ses.celldata.loc[iN,'rf_elevation']     = np.nanmedian(ses.celldata.loc[ses.celldata[idx_near].index,'rf_elevation'])
       
    return sessions

def filter_nearlabeled(ses,radius=50):

    if not hasattr(ses,'distmat_xyz'):
        [ses] = compute_pairwise_metrics([ses])
    temp = ses.distmat_xyz.copy()
    np.fill_diagonal(temp,0)  #this is to include the labeled neurons themselves
    closemat = temp[ses.celldata['redcell']==1,:] <= radius

    return np.any(closemat,axis=0)
"""
Modified on Sat May 2 2025

@author: Tilendra Choudhary
"""

"""

MULTIMODAL PHYSIOLOGY TOOLBOX (MPT): A comprehensive python-based toolbox for extracting physiomarkers from multimodal biosignals 

"""



'''
Example usage:
-------------

mpt_process = mpt.physiomarkers_from_biosignals(ecg=ecg_s, ppg=ppg_s, resp=resp_s, abp=abp_s, fs=fs, Show_plotly=False, Show_matplotlib=True) 

mpt_process.FiducialpointDetect_with_SQA(F_s=1000)
fiducials = mpt_process.fiducials 
signals_filtered = mpt_process.signals_filt
F_s = mpt_process.F_s #samples/seconds

physiomarkers_all = mpt_process.feature_extraction()
flat_dict = mpt_process.summary_statistics(stat_method='all')
df_physiomarkers = pd.DataFrame([flat_dict])

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, medfilt, find_peaks, filtfilt 
import resampy
import neurokit2 as nk
import MPT_plot_multiwav
import warnings
warnings.filterwarnings('ignore')
from scipy import spatial
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
import inspect
from scipy import interpolate
from sklearn.decomposition import PCA
import MPT_featureExtraction_utilities as utils
# import streamlit as st


class physiomarkers_from_biosignals():
    
    def __init__(self, ecg, ppg, resp, abp, fs, Show_plotly=True, Show_matplotlib=True, mse_scale=20):  
        self.ecg = ecg   # ECG signal 
        self.ppg = ppg   # PPG signal
        self.resp = resp # Respiratory signal
        self.abp = abp # Arterial blood pressure
        self.fs = fs  # Original sampling frequency
        self.Show_plotly = Show_plotly # Plotting flag for signals and fiducials via plotly
        self.Show_matplotlib = Show_matplotlib # Ploting flag for signals, fiducials and HRV-features via matplotlib
        self.mse_scale = mse_scale #Scale for MSE features
        

        
    """
    Denoising, filtering, flatling/clipping detection and SQI function

    """
    
    def FiducialpointDetect_with_SQA(self, F_s=1000):  #F_s: resampling frequency
        ecg = self.ecg
        ppg = self.ppg
        resp = self.resp
        abp = self.abp
        fs = self.fs
        figure_plot = self.Show_matplotlib
        self.F_s = F_s
            
        if figure_plot == True:
            fig, axs = plt.subplots(5, 1, figsize =(21, 9))
        
        sig_map  = {0:ecg, 1:ppg, 2:resp, 3:abp}
        Ls = np.max([len(ecg), len(ppg), len(resp), len(abp)])/fs  # Signal duration in seconds 
        
        if Ls == 0:
            raise TypeError("Please specify atleast one input signal.")
        else:
            sig_idx = np.argmax([len(ecg), len(ppg), len(resp), len(abp)])
            sig = sig_map[sig_idx]
            sig_resampled = resampy.resample(sig, fs, F_s)
            L_resample = len(sig_resampled)

       
        '''
        Various sequential operations are performed for each of the input raw signals, only if:
            (a) Signal is non-empty, i.e. len(signal) ! = 0,
            (b) Signal is non-flat for less than 80% of signal region
        
        They are executed in following stages:         
        - Stage-I: Flatline or clipping detection
        - Stage-II: Resampling all signals to F_s = 1000 Hz
        - Stage-III: Filtering process with outlier correction (Median filtering followed by band pass filtering)
        - Stage-IV: Estimation of Fiducial Points: Peak/troughs Detection and Correction Logic
        - Stage-V: Signal quality assessment (SQA) and validation of detected fiducials

        
        >>> Our SQA Criteria applied on ECG, PPG and ABP: 
        Our SQA decision is based on (i) template matching, and (ii) HR based physiological limit check.
        Note that template matching for PPG data looks from first detected foot (or onset) to last detected foot, if morphological
        method is provided, otherwise PPG peaks considered. While for ECG, its a symmetric window around every R-peak.
        
         (i) Criteria 1: Template matching based logic (Pearsorn's corr with an averaged template) 
        (ii) Criteria 2: HR based physiological limit check

        '''
        
        ix_start_stop1, flatline_loc1 = self.tc_flatline_boundry_detect(ecg, n=int(np.ceil(1*fs))) ### 1 seconds
        ix_start_stop2, flatline_loc2 = self.tc_flatline_boundry_detect(ppg, n=int(np.ceil(1*fs)))
        ix_start_stop3, flatline_loc3 = self.tc_flatline_boundry_detect(resp, n=int(np.ceil(1*fs)))
        ix_start_stop4, flatline_loc4 = self.tc_flatline_boundry_detect(abp, n=int(np.ceil(1*fs)))

        flatline = {'flat_loc_ecg':flatline_loc1, 'flat_loc_ppg':flatline_loc2, 'flat_loc_resp':flatline_loc3, 'flat_loc_abp':flatline_loc4}
        self.flatline = flatline

        Ks = int(np.floor(0.015*F_s))#Median filtering window to reduce irregularity and intermittency (15 ms)
        
        try:
            if len(ecg) != 0 and np.sum(flatline_loc1)<0.80*len(flatline_loc1):  
                ecg[(flatline_loc1==1)] = np.median(ecg)
                ecg = self.signal_oulier_correction(ecg, th=6)
                ecg = resampy.resample(ecg, fs, F_s)
                ecg_1 = medfilt(ecg, kernel_size=Ks) 
                ecg_filtered = self.final_filter(ecg_1, F_s, cutoff_high=0.8, cutoff_low=60, order=4)
                ecg_filtered = self.signal_oulier_correction(ecg_filtered, th=5)               
                R_flag, info = nk.ecg_peaks(ecg_filtered, sampling_rate=F_s, method="neurokit")
                Rpks = info['ECG_R_Peaks']
                # Rtrue = self.Peak_correction(Rpks, ecg_filtered, F_s, peaktype=1) ###--- Peak Correction within a search window----###
                # Rtrue = self.point_outlier_rejection(ecg_filtered, Rtrue) #sanitization of detected fiducial points
                ecgDeli_feat_dict, Rtrue = self.ECG_features_delineate(ecg_filtered, F_s, info) #Performs SQI and delineation
                Rtrue = Rtrue[1:-1] #To avaid start and end effects after processing, one peak from start and end locations are discarded.
                R_flag_true = pd.Series(np.zeros(len(ecg_filtered)), name='ECG_R_Peaks')
                R_flag_true.iloc[Rtrue] = 1  ### Binarization to show rpeaks location in ECG signal
                R_flag_true = R_flag_true.astype(int)
                #self.R_flag_true = R_flag_true
                ### EDR: ECG Derived Respiration and other features--------------------------------------
                hb_nor = []
                if len(Rtrue)>2:
                    rpeaks = pd.DataFrame(R_flag_true)
                    ecg_rate = nk.signal_rate(rpeaks, sampling_rate=F_s, desired_length=len(rpeaks))
                    edr = nk.ecg_rsp(ecg_rate, sampling_rate=F_s)
                    edr = nk.rsp_clean(edr, sampling_rate=F_s)
                    edr_filtered = self.signal_oulier_correction(edr, th=5)
                    edr_e = find_peaks(edr_filtered, distance=2*fs, width=0.5*fs)[0] #prominence=0.0001,
                    edr_e = pd.Series(edr_e[1:-1])
                    edr_i = pd.Series(self.PPG_foot(edr_e, edr_filtered))
                    ## Heartbeat normalization and stacking
                    for k in range(1, len(Rtrue)-1):
                        heartbeat_ik = ecg_filtered[int(Rtrue[k]-250e-3*F_s):int(Rtrue[k+1]-250e-3*F_s)]
                        #Period normalization to 1*F_s and stacking
                        f = interpolate.interp1d(np.arange(0, len(heartbeat_ik)), heartbeat_ik, kind='cubic')
                        xnew = np.linspace(0, len(heartbeat_ik)-1, num=1*F_s, endpoint=True)
                        hb_nor_ik = f(xnew)
                        hb_nor.append(hb_nor_ik)
                else: edr_filtered, edr_i, edr_e = pd.Series([]), pd.Series([]), pd.Series([])
                ecgHB_feat_dict = {}
                if len(hb_nor)>0:
                    hb_i_all = np.array(hb_nor).T
                    #PCA
                    pca_hb = PCA(n_components=3)
                    pcs = pca_hb.fit_transform(hb_i_all)
                    hb_pc1 = pcs[:,0]
                    #make beat zero mean and amplitude normalized
                    hb_pc1 = hb_pc1 - np.mean(hb_pc1)
                    hb_pc1 = hb_pc1/np.max(np.abs(hb_pc1))
                    eig_val_hb = pca_hb.explained_variance_
                    interbeat_eigenvalues = eig_val_hb[:3] #top 3
                    center_freq = self.signal_center_frequency(hb_pc1, F_s, show=False)
                    ecgHB_feat_dict['ecgBeatCenterFreq']=center_freq
                    ecgHB_feat_dict['ecgInterbeat_eigval1']=interbeat_eigenvalues[0]
                    ecgHB_feat_dict['ecgInterbeat_eigval2']=interbeat_eigenvalues[1]
                    ecgHB_feat_dict['ecgInterbeat_eigval3']=interbeat_eigenvalues[2]
                else:
                    ecgHB_feat_dict['ecgBeatCenterFreq']=np.nan
                    ecgHB_feat_dict['ecgInterbeat_eigval1']=np.nan
                    ecgHB_feat_dict['ecgInterbeat_eigval2']=np.nan
                    ecgHB_feat_dict['ecgInterbeat_eigval3']=np.nan

                Rtrue_final = pd.Series(Rtrue)
                ecg_HRn = 60*F_s/Rtrue_final.diff()
                R_f = Rtrue_final.copy()
                
                length = len(ecg)
                T = (length - 1) / F_s
                t = np.linspace(0, T, length, endpoint=False)
                if figure_plot == True:
                    axs[0].plot(t,ecg_filtered)
                    axs[0].scatter(t[R_f],ecg_filtered[R_f], c='red')
                    axs[0].set_ylabel('ECG')
                    axs[0].autoscale(enable=True, axis='x', tight=True)
                    axs[0].set_title('Preprocessed waveforms')

                    axs[3].plot(t,edr_filtered)
                    axs[3].set_ylabel('EDR Resp.')
                    axs[3].scatter(t[edr_i],edr_filtered[edr_i], c='red')
                    axs[3].scatter(t[edr_e],edr_filtered[edr_e],c='green')
                    axs[3].autoscale(enable=True, axis='x', tight=True)
            else: 
                ecg_filtered = np.zeros(L_resample) 
                edr_filtered = np.zeros(L_resample) 
                Rtrue_final, ecg_HRn, edr_i, edr_e, = pd.Series([]), pd.Series([]), pd.Series([]), pd.Series([])
                keys = ['atrialSys_phase', 'ventSys_phase', 'P_duration', 'QRS_duration', 'T_duration', 'pr_interval_ms',
                         'pr_segment_ms', 'st_interval_ms', 'st_segment_ms', 'qt_interval_ms', 'qtc_interval_ms']
                ecgDeli_feat_dict = {key:np.nan for key in keys}
                keys = ['ecgBeatCenterFreq', 'ecgInterbeat_eigval1', 'ecgInterbeat_eigval2', 'ecgInterbeat_eigval3']
                ecgHB_feat_dict = {key:np.nan for key in keys}
        except (IndexError, ValueError) as e: 
            ecg_filtered = np.zeros(L_resample) 
            edr_filtered = np.zeros(L_resample) 
            Rtrue_final, ecg_HRn, edr_i, edr_e, = pd.Series([]), pd.Series([]), pd.Series([]), pd.Series([])
            keys = ['atrialSys_phase', 'ventSys_phase', 'P_duration', 'QRS_duration', 'T_duration', 'pr_interval_ms',
                        'pr_segment_ms', 'st_interval_ms', 'st_segment_ms', 'qt_interval_ms', 'qtc_interval_ms']
            ecgDeli_feat_dict = {key:np.nan for key in keys}
            keys = ['ecgBeatCenterFreq', 'ecgInterbeat_eigval1', 'ecgInterbeat_eigval2', 'ecgInterbeat_eigval3']
            ecgHB_feat_dict = {key:np.nan for key in keys}

        self.ecgDeli_feat_dict = ecgDeli_feat_dict
        self.ecgHB_feat_dict = ecgHB_feat_dict

        try:
            if len(ppg) != 0 and np.sum(flatline_loc2)<0.80*len(flatline_loc2): 
                ppg[(flatline_loc2==1)] = np.median(ppg)
                ppg = self.signal_oulier_correction(ppg, th=6)
                ppg = resampy.resample(ppg, fs, F_s)
                ppg_1 = medfilt(ppg, kernel_size=Ks)
                ppg_filtered = self.final_filter(ppg_1, F_s, cutoff_high=0.5, cutoff_low=8, order=2)
                ppg_filtered = self.signal_oulier_correction(ppg_filtered, th=5)
                # PPG peaks: Delta parameter can be adjusted according to the amplitude of the signal.
                ###--- PPG Peaks Detection ----### method (Elgendi et al., 2013)
                peaks, info = nk.ppg_peaks(ppg_filtered, sampling_rate=F_s, method="elgendi", correct_artifacts=True, show=False)
                Ptrue0 = info['PPG_Peaks'] # Systolic peak (SP) of PPG
                Ptrue,_,_,_ = self.cardiac_SQI_beats(ppg_filtered, F_s, Ptrue0) #Performs SQI
                Ptrue_final = pd.Series(Ptrue[1:-1])
                # Ptrue = self.point_outlier_rejection(ppg_filtered, Ptrue0) #sanitization of detected fiducial points
                ###--- PPG Foot (Onset) Detection ----###
                foot0 = self.PPG_foot(Ptrue_final, ppg_filtered)
                foot_final = pd.Series(self.Peak_correction(foot0, ppg_filtered, F_s, peaktype=0)) # onset/offset or foot of PPG
                ### PDR: PPG Derived Respiration--------------------------------------
                hb_nor = []
                if len(Ptrue_final)>2:
                    ppgpeaks = pd.Series(np.zeros(len(ppg_filtered)), name='PPG_SP_Peaks')
                    ppgpeaks.iloc[Ptrue_final] = 1  ### Binarization to show rpeaks location in PPG signal
                    ppgpeaks = pd.DataFrame(ppgpeaks)
                    ppg_rate = nk.signal_rate(ppgpeaks, sampling_rate=F_s, desired_length=len(ppgpeaks))
                    pdr = nk.ecg_rsp(ppg_rate, sampling_rate=F_s)
                    pdr = nk.rsp_clean(pdr, sampling_rate=F_s)
                    pdr_filtered = self.signal_oulier_correction(pdr, th=5)
                    pdr_e = find_peaks(pdr_filtered, distance=2*fs, width=0.5*fs)[0] #prominence=0.0001,
                    pdr_e = pd.Series(pdr_e[1:-1])
                    pdr_i = pd.Series(self.PPG_foot(pdr_e, pdr_filtered))
                    ## Heartbeat normalization and stacking
                    for k in range(1, len(Ptrue_final)-1):
                        heartbeat_ik = ppg_filtered[int(Ptrue_final[k]-250e-3*F_s):int(Ptrue_final[k+1]-250e-3*F_s)]
                        #Period normalization to 1*F_s and stacking
                        f = interpolate.interp1d(np.arange(0, len(heartbeat_ik)), heartbeat_ik, kind='cubic')
                        xnew = np.linspace(0, len(heartbeat_ik)-1, num=1*F_s, endpoint=True)
                        hb_nor_ik = f(xnew)
                        hb_nor.append(hb_nor_ik)
                else:
                    pdr_filtered = np.zeros(L_resample) 
                    pdr_i, pdr_e, = pd.Series([]), pd.Series([])
                ppgHB_feat_dict = {}
                if len(hb_nor)>0:
                    hb_i_all = np.array(hb_nor).T
                    #PCA
                    pca_hb = PCA(n_components=3)
                    pcs = pca_hb.fit_transform(hb_i_all)
                    hb_pc1 = pcs[:,0]
                    #make beat zero mean and amplitude normalized
                    hb_pc1 = hb_pc1 - np.mean(hb_pc1)
                    hb_pc1 = hb_pc1/np.max(np.abs(hb_pc1))
                    eig_val_hb = pca_hb.explained_variance_
                    interbeat_eigenvalues = eig_val_hb[:3] #top 3
                    center_freq = self.signal_center_frequency(hb_pc1, F_s, show=False)
                    ppgHB_feat_dict['ppgBeatCenterFreq']=center_freq
                    ppgHB_feat_dict['ppgInterbeat_eigval1']=interbeat_eigenvalues[0]
                    ppgHB_feat_dict['ppgInterbeat_eigval2']=interbeat_eigenvalues[1]
                    ppgHB_feat_dict['ppgInterbeat_eigval3']=interbeat_eigenvalues[2]
                else:
                    ppgHB_feat_dict['ppgBeatCenterFreq']=np.nan
                    ppgHB_feat_dict['ppgInterbeat_eigval1']=np.nan
                    ppgHB_feat_dict['ppgInterbeat_eigval2']=np.nan
                    ppgHB_feat_dict['ppgInterbeat_eigval3']=np.nan



                ppg_HRn = 60*F_s/Ptrue_final.diff()
                locs_peaks, PPG_peak_amps, locs_onsets, PPG_onset_amps, vpg_signal, apg_signal, vpg_fiducials, apg_fiducials, ppg_fiducials = self.ppg_fiducials_detection(ppg_filtered, F_s, Ptrue_final, foot_final)
                P_f = Ptrue_final.copy()
                F_f = foot_final.copy()
                length = len(ppg)
                T = (length - 1) / F_s
                t = np.linspace(0, T, length, endpoint=False)
                if figure_plot:           
                    axs[1].plot(t,ppg_filtered)
                    axs[1].scatter(t[P_f],ppg_filtered[P_f], c='red')
                    axs[1].scatter(t[F_f],ppg_filtered[F_f],c='green')
                    axs[1].set_ylabel('PPG')
                    axs[1].autoscale(enable=True, axis='x', tight=True)

                    fig, ax = plt.subplots(3,1, figsize=(21,6.5))  #(8,4)
                    ax[0].plot(t, ppg_filtered)
                    if len(Ptrue_final)>0: ax[0].scatter(t[Ptrue_final],ppg_filtered[Ptrue_final], c='red')
                    if len(foot_final)>0: ax[0].scatter(t[foot_final],ppg_filtered[foot_final],c='green')
                    if len(ppg_fiducials['D_waves'])>0: ax[0].scatter(t[ppg_fiducials['D_waves']],ppg_filtered[ppg_fiducials['D_waves']], c='black')
                    if len(ppg_fiducials['N_waves'])>0: ax[0].scatter(t[ppg_fiducials['N_waves']],ppg_filtered[ppg_fiducials['N_waves']], c='magenta')
                    ax[0].set_ylabel('PPG')
                    ax[0].set_xlim([0,Ls])


                    ax[1].plot(t, vpg_signal)
                    if len(vpg_fiducials['w_waves'])>0: ax[1].scatter(t[vpg_fiducials['w_waves']],vpg_signal[vpg_fiducials['w_waves']], c='red')
                    if len(vpg_fiducials['y_waves'])>0: ax[1].scatter(t[vpg_fiducials['y_waves']],vpg_signal[vpg_fiducials['y_waves']],c='green')
                    if len(vpg_fiducials['z_waves'])>0: ax[1].scatter(t[vpg_fiducials['z_waves']],vpg_signal[vpg_fiducials['z_waves']],c='black')
                    ax[1].set_ylabel('VPG')
                    ax[1].set_xlim([0,Ls])

                    ax[2].plot(t, apg_signal)
                    if len(apg_fiducials['a_waves'])>0: ax[2].scatter(t[apg_fiducials['a_waves']],apg_signal[apg_fiducials['a_waves']], c='red')
                    if len(apg_fiducials['b_waves'])>0: ax[2].scatter(t[apg_fiducials['b_waves']],apg_signal[apg_fiducials['b_waves']],c='green')
                    if len(apg_fiducials['e_waves'])>0: ax[2].scatter(t[apg_fiducials['e_waves']],apg_signal[apg_fiducials['e_waves']],c='black')
                    ax[2].set_ylabel('APG')
                    ax[2].set_xlabel('Time (s)')
                    ax[2].set_xlim([0,Ls])

            else: 
                vpg_fiducials = {'w_waves':[],'y_waves':[], 'z_waves':[]}
                apg_fiducials = {'a_waves':[], 'b_waves':[], 'c_waves':[], 'd_waves':[], 'e_waves':[]}
                ppg_fiducials = {'O_waves':[], 'S_waves':[], 'N_waves':[], 'D_waves':[]}
                Ptrue_final, foot_final, ppg_HRn, apg_signal = pd.Series([]), pd.Series([]), pd.Series([]), pd.Series([])
                ppg_filtered = np.zeros(L_resample) 
                vpg_signal = np.zeros(L_resample)
                apg_signal = np.zeros(L_resample) 
                pdr_filtered = np.zeros(L_resample) 
                pdr_i, pdr_e, = pd.Series([]), pd.Series([])
                keys = ['ppgBeatCenterFreq', 'ppgInterbeat_eigval1', 'ppgInterbeat_eigval2', 'ppgInterbeat_eigval3']
                ppgHB_feat_dict = {key:np.nan for key in keys}
        except (IndexError, ValueError) as e:  
            vpg_fiducials = {'w_waves':[],'y_waves':[], 'z_waves':[]}
            apg_fiducials = {'a_waves':[], 'b_waves':[], 'c_waves':[], 'd_waves':[], 'e_waves':[]}
            ppg_fiducials = {'O_waves':[], 'S_waves':[], 'N_waves':[], 'D_waves':[]}
            Ptrue_final, foot_final, ppg_HRn, apg_signal = pd.Series([]), pd.Series([]), pd.Series([]), pd.Series([])
            ppg_filtered = np.zeros(L_resample) 
            vpg_signal = np.zeros(L_resample)
            apg_signal = np.zeros(L_resample) 
            pdr_filtered = np.zeros(L_resample) 
            pdr_i, pdr_e, = pd.Series([]), pd.Series([])
            keys = ['ppgBeatCenterFreq', 'ppgInterbeat_eigval1', 'ppgInterbeat_eigval2', 'ppgInterbeat_eigval3']
            ppgHB_feat_dict = {key:np.nan for key in keys}
        
        self.ppgHB_feat_dict = ppgHB_feat_dict
            
        try:
            if len(resp) != 0 and np.sum(flatline_loc3)<0.80*len(flatline_loc3): 
                resp[(flatline_loc3==1)] = np.median(resp)
                resp = self.signal_oulier_correction(resp, th=6)
                resp = resampy.resample(resp, fs, F_s)
                resp_1 = medfilt(resp, kernel_size=Ks)
                resp_filtered = self.final_filter(resp_1, F_s, cutoff_high=0.1, cutoff_low=0.5, order=2)
                # resp_filtered = nk.rsp_clean(resp_filtered, sampling_rate=F_s)
                ####Resp: [0.1 0.5] Hz corresponding to 6–30 Breaths/min. Ref: https://doi.org/10.3390%2Fs18113705
                resp_filtered = self.signal_oulier_correction(resp_filtered, th=5)
                ###--- Respiratory extrema and phases (inspiratory and expiratory)
                # p_i (troughs) are p_e (peaks) are onsets for inhalation and exhalation, respectively.
                p_e=find_peaks(resp_filtered, distance=2*F_s, width=0.5*F_s)[0] #prominence=0.0001,
                p_e = pd.Series(p_e[1:-1])
                p_i = pd.Series(self.PPG_foot(p_e, resp_filtered))
                length = len(resp)
                T = (length - 1) / F_s
                t = np.linspace(0, T, length, endpoint=False)
                if figure_plot == True: 
                    axs[2].plot(t,resp_filtered)
                    axs[2].set_ylabel('Resp.')
                    axs[2].scatter(t[p_i],resp_filtered[p_i], c='red')
                    axs[2].scatter(t[p_e],resp_filtered[p_e],c='green')
                    axs[2].autoscale(enable=True, axis='x', tight=True)
            else: 
                p_i, p_e = pd.Series([]), pd.Series([])
                resp_filtered = np.zeros(L_resample) 
        except (IndexError, ValueError) as e: 
            p_i, p_e = pd.Series([]), pd.Series([])
            resp_filtered = np.zeros(L_resample) 
            
        
        try:
            if len(abp) != 0 and np.sum(flatline_loc4)<0.80*len(flatline_loc4): 
                abp[(flatline_loc4==1)] = np.median(abp)
                abp = self.signal_oulier_correction(abp, th=6)
                abp = resampy.resample(abp, fs, F_s)
                abp_filtered = medfilt(abp, kernel_size=Ks)
                d, c = self.butter_lowpass(cutoff=15, fs=F_s, order=4) #Low pass filtering
                abp_filtered = filtfilt(d, c, abp_filtered)
                abp_filtered = self.signal_oulier_correction(abp_filtered, th=5)
                SBPtrue0 = find_peaks(abp_filtered, distance=0.350*F_s, height=50)[0]
                SBPtrue,_,_,_ = self.cardiac_SQI_beats(abp_filtered, F_s, SBPtrue0) #Performs SQI
                SBP_final = pd.Series(SBPtrue[1:-1])
                # SBPtrue = self.point_outlier_rejection(abp_filtered, SBPtrue) #sanitization of detected fiducial points
                ###--- DBP (Onset) Detection ----###
                DBP0 = self.PPG_foot(SBPtrue, abp_filtered)
                DBP_final = pd.Series(self.Peak_correction(DBP0, abp_filtered, F_s, peaktype=0)) # onset/offset for DBP troughs
                length = len(abp)
                T = (length - 1) / F_s
                t = np.linspace(0, T, length, endpoint=False)
                if figure_plot == True:           
                    axs[4].plot(t,abp_filtered)
                    axs[4].scatter(t[SBP_final],abp_filtered[SBP_final], c='red')
                    axs[4].scatter(t[DBP_final],abp_filtered[DBP_final],c='green')
                    axs[4].set_ylabel('ABP')
                    axs[4].autoscale(enable=True, axis='x', tight=True)
                    axs[4].set_xlabel('Time (s)')
            else: 
                SBP_final, DBP_final = pd.Series([]), pd.Series([])
                abp_filtered = np.zeros(L_resample) 
        except (IndexError, ValueError) as e: 
            SBP_final, DBP_final = pd.Series([]), pd.Series([])
            abp_filtered = np.zeros(L_resample) 
            
        
        
        
        if self.Show_plotly:
            signals = {}
            peaks = {}
            if (ecg_filtered==0).all() == False:
                signals['ECG'] = ecg_filtered
                peaks['ECG'] = {'R-peak': R_f}
            if (ppg_filtered==0).all() == False:
                signals['PPG'] = ppg_filtered
                peaks['PPG'] = {'systolic peak (SP)': P_f,'onset or offset (O)': F_f,
                       'dicrotic notch (DN)':ppg_fiducials['N_waves'], 'diastolic peak (DP)':ppg_fiducials['D_waves']}
            if (resp_filtered==0).all() == False:
                signals['Resp.'] = resp_filtered
                peaks['Resp.'] = {'inspiration start':p_i,'expiration start': p_e}
            if (edr_filtered==0).all() == False:
                signals['EDR'] = edr_filtered
                peaks['EDR'] = {'inspiration start':edr_i,'expiration start': edr_e}
            if (abp_filtered==0).all() == False:
                signals['ABP'] = abp_filtered
                peaks['ABP'] = {'SBP':SBP_final,'DBP': DBP_final}
           
            MPT_plot_multiwav.MPT_plot_multiwav(signals,peaks,method='plotly',timestamps=t, timestamp_resolution='s',width=1000,height=440,show_peaks=True)

        fiducials = {'Rpeak':Rtrue_final, 'PPGpeak':Ptrue_final, 'PPGfoot':foot_final, 'Resp_i':p_i, 'Resp_e':p_e, 'EDR_i':edr_i, 'EDR_e':edr_e, \
                     'ppg_fiducials':ppg_fiducials, 'vpg_fiducials':vpg_fiducials, 'apg_fiducials':apg_fiducials, 'ecg_HR':ecg_HRn, 'ppg_HR':ppg_HRn, \
                     'SBP_ix': SBP_final, 'DBP_ix': DBP_final, 'PDR_i':pdr_i, 'PDR_e':pdr_e}
        

        signals_filt = pd.DataFrame({'ecg': ecg_filtered, 'ppg':ppg_filtered, 'resp':resp_filtered, 'vpg':vpg_signal, 'apg': apg_signal, 'edr': edr_filtered, 'abp': abp_filtered, 'pdr': pdr_filtered})
        
        self.signals_filt = signals_filt
        self.fiducials = fiducials
        
        
        
    
    ###################################################################

    

    ################# STAGE III: Extracting features by calling functions #############################################3
    def feature_extraction(self):
        F_s = self.F_s
        signals_filt = self.signals_filt
        fiducials = self.fiducials
        
        ############## Signals ################################
        ppg_signal = signals_filt['ppg'].to_numpy() #ppg signal
        vpg_signal = signals_filt['vpg'] #VPG signal
        abp_signal = signals_filt['abp'] #ABP signal
        ecg_signal = signals_filt['ecg'] #ECG signal
        
        ################# Fiducials ########################### 
        Rpeak = fiducials['Rpeak'].astype(int) #samples
        Resp_i = fiducials['Resp_i'].astype(int) #samples
        Resp_e = fiducials['Resp_e'].astype(int) #samples
        edr_i = fiducials['EDR_i'].astype(int) #samples
        edr_e = fiducials['EDR_e'].astype(int) #samples
        pdr_i = fiducials['PDR_i'].astype(int) #samples
        pdr_e = fiducials['PDR_e'].astype(int) #samples
        
        PPGpeak = fiducials['PPGpeak'].astype(int) #samples
        PPGfoot = fiducials['PPGfoot'].astype(int) #samples
        vpg_fiducials = fiducials['vpg_fiducials']
        apg_fiducials = fiducials['apg_fiducials']
        ppg_fiducials = fiducials['ppg_fiducials']
        #ppg waves
        D_waves = ppg_fiducials.get('D_waves')
        N_waves = ppg_fiducials.get('N_waves')
        #vpg waves
        w_waves = vpg_fiducials.get('w_waves')
        #x_waves = vpg_fiducials.get('x_waves')
        
        
        SBP_final = fiducials['SBP_ix'].astype(int) #samples
        DBP_final = fiducials['DBP_ix'].astype(int) #samples
        
    
        
        ecg_Rpeaks = Rpeak.values
        locs_peaks = PPGpeak.values
        locs_onsets = PPGfoot.values
        
    
    
        ###--- RR, PP, FF interval -----###
        RRecg = pd.Series(Rpeak.diff()[1:].values)
        PPppg = pd.Series(PPGpeak.diff()[1:].values)
    
 
    
        RRecg = (RRecg*1000/F_s) #in ms
        PPppg = (PPppg*1000/F_s) #in ms
        # FFppg = (FFppg*1000/F_s) #in ms
        
        PPedr = np.diff(edr_i)/F_s #in s
        

        
    
        ###----- Medians of all Features -----###
        RRecg_med = RRecg.median()  #in ms
        PPppg_med = PPppg.median()
        # FFppg_med = FFppg.median()
        # PPedr_med = pd.Series(PPedr).median() #in s
    
        
        
        ###### ------- HRV PARAMETER ESTIMATION FROM ECG (FROM PPG, IF ECG IS UNAVAILABLE) -----#####
        
        #if ECG available and <=10% of RRecgs are beyond range 300-2000ms
        if (len(RRecg) != 0) and ((100*len(np.where((RRecg>2000) | (RRecg<300))[0])/len(RRecg)) <= 20):
            NN = RRecg.copy()
            Npks = Rpeak.copy()
            pks_label = 'ECG_R_Peaks'
            cardiac_signal = ecg_signal.copy()
            Availability_flag = True
        #if ECG available and >10% of RRecgs are beyond range 300-2000ms
        elif (len(RRecg) != 0) and ((100*len(np.where((RRecg>2000) | (RRecg<300))[0])/len(RRecg)) > 20) and (len(PPppg) != 0):
            NN = PPppg.copy()
            Npks = PPGpeak.copy()
            pks_label = 'PPG_Peaks'
            cardiac_signal = ppg_signal.copy()
            Availability_flag = True 
        #if ECG or RRecg is unavailable 
        elif (len(RRecg) == 0) and (len(PPppg) != 0):
            NN = PPppg.copy()
            Npks = PPGpeak.copy()
            pks_label = 'PPG_Peaks'
            cardiac_signal = ppg_signal.copy()
            Availability_flag = True    
        else: 
            Availability_flag = False
            msen_AUC_nu = np.nan 
            MSE_series = np.nan * np.ones(self.mse_scale)
            MSE_dict = {} 
            for m in range(len(MSE_series)): MSE_dict[f'MSE_{m}'] = MSE_series[m] 
            hrv_RecurrenceRate, hrv_Determinism, hrv_Laminarity = np.nan*np.ones(3)
            keys = [
                    'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 'HRV_SDNNI2',
                    'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD',
                    'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD',
                    'HRV_Prc20NN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN',
                    'HRV_HTI', 'HRV_TINN', 'HRV_ULF', 'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF',
                    'HRV_TP', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2',
                    'HRV_SD1SD2', 'HRV_S', 'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP',
                    'HRV_IALS', 'HRV_PSS', 'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI',
                    'HRV_C1d', 'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d',
                    'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_DFA_alpha1',
                    'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak', 'HRV_MFDFA_alpha1_Mean',
                    'HRV_MFDFA_alpha1_Max', 'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry',
                    'HRV_MFDFA_alpha1_Fluctuation', 'HRV_MFDFA_alpha1_Increment', 'HRV_DFA_alpha2',
                    'HRV_MFDFA_alpha2_Width', 'HRV_MFDFA_alpha2_Peak', 'HRV_MFDFA_alpha2_Mean',
                    'HRV_MFDFA_alpha2_Max', 'HRV_MFDFA_alpha2_Delta', 'HRV_MFDFA_alpha2_Asymmetry',
                    'HRV_MFDFA_alpha2_Fluctuation', 'HRV_MFDFA_alpha2_Increment', 'HRV_ApEn',
                    'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn',
                    'HRV_RCMSEn', 'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC'
                ]
            hrv_dict = {key: np.nan for key in keys}
    
                
        if Availability_flag == True:
            
            try:
                # ''' Outlier correction '''           
                ##-------------------------------------------------------
                # This remove outliers from signal
                NN_intervals_without_outliers = remove_outliers(rr_intervals=NN.values, low_rri=300, high_rri=2000, verbose=False)
                # This replace outliers nan values with linear interpolation
                interpolated_NN_intervals = interpolate_nan_values(rr_intervals=NN_intervals_without_outliers,interpolation_method="linear")
                # This remove ectopic beats from signal
                NN_no_ectopic_beats = remove_ectopic_beats(rr_intervals=interpolated_NN_intervals, method="malik", verbose=False)
                # This replace ectopic beats nan values with linear interpolation
                NN_filtered = np.asarray(np.ceil(interpolate_nan_values(rr_intervals=NN_no_ectopic_beats)), dtype='int')
                ##-------------------------------------------------------
            except:
                NN_filtered = NN.values
     
            if self.Show_matplotlib:
                fig, ax = plt.subplots(2, 1)
                ax[0].plot(NN)
                ax[0].set_ylabel('Before filtering')
                ax[1].plot(NN_filtered)
                ax[1].set_ylabel('After filtering')
            
            self.fiducials['NN'] = NN_filtered
            
            if self.Show_matplotlib:
                plt.figure()

            Npks_flag = pd.Series(np.zeros(len(cardiac_signal)), name=pks_label)
            Npks_flag.iloc[Npks] = 1  ### Binarization to show peaks location in cardiac signal ecg/ppg
            Npks_flag = Npks_flag.astype(int)
            
            
            #hrv_indices = nk.hrv(Npks_flag, sampling_rate=F_s, show=True)
            #st.pyplot(plt, clear_figure=True)
            #hrv_dict = hrv_indices.to_dict(orient='records')[0]


            # """ HRV - Time domain analysis """
            hrv_t = nk.hrv_time(Npks_flag, sampling_rate=F_s, show=False)
            hrv_t = hrv_t.to_dict(orient='records')[0]

            hrv_f = nk.hrv_frequency(Npks_flag, sampling_rate=F_s, show=False)
            hrv_f = hrv_f.to_dict(orient='records')[0]

            try:
                hrv_nl = nk.hrv_nonlinear(Npks_flag, sampling_rate=F_s, show=False)
                hrv_nl = hrv_nl.to_dict(orient='records')[0]
            except ValueError as e:  
                nl_keys = [
                    'HRV_SD1', 'HRV_SD2',
                    'HRV_SD1SD2', 'HRV_S', 'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP',
                    'HRV_IALS', 'HRV_PSS', 'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI',
                    'HRV_C1d', 'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d',
                    'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_DFA_alpha1',
                    'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak', 'HRV_MFDFA_alpha1_Mean',
                    'HRV_MFDFA_alpha1_Max', 'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry',
                    'HRV_MFDFA_alpha1_Fluctuation', 'HRV_MFDFA_alpha1_Increment', 'HRV_DFA_alpha2',
                    'HRV_MFDFA_alpha2_Width', 'HRV_MFDFA_alpha2_Peak', 'HRV_MFDFA_alpha2_Mean',
                    'HRV_MFDFA_alpha2_Max', 'HRV_MFDFA_alpha2_Delta', 'HRV_MFDFA_alpha2_Asymmetry',
                    'HRV_MFDFA_alpha2_Fluctuation', 'HRV_MFDFA_alpha2_Increment', 'HRV_ApEn',
                    'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn',
                    'HRV_RCMSEn', 'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC'
                    ]
                hrv_nl = {key: np.nan for key in nl_keys}
            
            hrv_dict = hrv_t.copy()
            hrv_dict.update(hrv_f)
            hrv_dict.update(hrv_nl)


       
            # """ HRV - DFA, Credit: Neurokit2. Detrended fluctuation analysis of HRV"""
            # try: DFA_alpha1, DFA_alpha2 = self.hrv_dfa_calculation(NN_filtered, show_flag=self.Show_matplotlib)
            # except ValueError: DFA_alpha1, DFA_alpha2 = np.nan, np.nan
            
            
            """ HRV - Multiscale Entropy (Complexity analysis) """            
            
            # if len(NN_filtered) > 80:
            
            
            try:
                msen_AUC_nu, info = nk.entropy_multiscale(NN_filtered, scale=self.mse_scale, method='MSEn', show=self.Show_matplotlib)
                MSE_series = info['Value']
            except Exception as e:
                msen_AUC_nu = np.nan
                MSE_series = np.nan * np.ones(self.mse_scale)

            MSE_dict = {}  # Create an empty dictionary to store values
            for m in range(len(MSE_series)):
                MSE_dict[f'MSE_{m}'] = MSE_series[m]  # Assign MSE_series[m] to the corresponding key in the dictionary
                 
            
            # """ HRV- heart rate asymmetry analysis (HRA) """
            # # Ref: Piskorski (2011), Asymmetric properties of long-term and total heart rate variability
            # #contributions of heart rate decelerations and accelerations to short- and long-term HRV 
            # C1_d, C1_a, SD1_d, SD1_a, C2_d, C2_a, SD2_d, SD2_a = self.hra_acc_dec(NN_filtered)

            
            """ HRV - Recurrence quantification analysis (RQA) """
            # Ref:
            # ----------
            # * Zimatore, G., Falcioni, L., Gallotta, M. C., Bonavolontà, V., Campanella, M., De Spirito, M.,
            #   ... & Baldari, C. (2021). Recurrence quantification analysis of heart rate variability to
            #   detect both ventilatory thresholds. PloS one, 16(10), e0249504.
            # * Ding, H., Crozier, S., & Wilson, S. (2008). Optimization of Euclidean distance threshold in
            #   the application of recurrence quantification analysis to heart rate variability studies.
            #   Chaos, Solitons & Fractals, 38(5), 1457-1467.
            
            
            try:
                # Linear detrend and tolerance selection (Zimatore, 2021)
                NN_filt_detrended = nk.signal.signal_detrend(NN_filtered, method="polynomial", order=1)
                # Radius (50% of mean distance between all pairs of points in time)
                dists = spatial.distance.pdist(np.array([NN_filt_detrended, NN_filt_detrended]).T, "euclidean")
                tolerance = 0.5 * np.mean(dists)
                # Run the RQA
                hrv_rqa, _ = nk.complexity_rqa(NN_filt_detrended,
                    dimension=7,
                    delay=1,
                    tolerance=tolerance,
                    show=self.Show_matplotlib)
                plt.show()
                hrv_RecurrenceRate = hrv_rqa['RecurrenceRate'][0]
                hrv_Determinism = hrv_rqa['Determinism'][0]
                hrv_Laminarity = hrv_rqa['Laminarity'][0]
            except ValueError: hrv_RecurrenceRate, hrv_Determinism, hrv_Laminarity = np.nan*np.ones(3)
        
    
    
        if signals_filt['ppg'].sum() != 0:
            AUCos_N, AUCso_N, T_os_f, T_so_f, T_so_cd, A_AC, A_sp, A_off, mean_slope_os, mean_slope_so, PW_10, PW_25, PW_33, PW_50, PW_66, PW_75 = self.AUCos_AUCso_area_time(ppg_signal, PPGfoot, PPGpeak, F_s, PPppg_med, vpg_signal)
            ACppg_med = np.median(A_AC)
            systolic_areas, diastolic_areas, IPA, T_sys, T_dias, DN_exists, A_dn = self.systolic_diastolic_area_time(ppg_signal, PPGfoot, N_waves, F_s, PPppg_med, ACppg_med)
            AUCow_N, AUCwo_N, T_ow_f, T_wo_f = self.AUCow_AUCwo_area_time(ppg_signal, PPGfoot, w_waves, F_s, PPppg_med, ACppg_med)
            pat, dpat = self.pulse_arrival_time(ecg_Rpeaks, locs_onsets, F_s, PPppg_med)
            PW_ms = self.pulse_width(locs_onsets, F_s, PPppg_med)
            pulse_areas = self.pulse_area(ppg_signal, PPGfoot, F_s, PPppg_med, ACppg_med) 
            pulse_int_ms, pulse_rate = self.pulse_rate_ppg(locs_peaks, F_s, PPppg_med)
            Pamp_diff_ppg = self.ppg_peak_amp_difference(ppg_signal, locs_peaks)
            Delta_T_sd, RI, Delta_A_dn_dp, T_dn_dp_ms, SI = self.ppg_systole_diastole_delineation(ppg_signal, PPGfoot, PPGpeak, D_waves, N_waves, F_s, PPppg_med)
            max_upslope = self.maximum_upslope_ppg(vpg_signal, w_waves)
            Aug_index = self.augmentation_index_ppg(ppg_signal, PPGpeak, PPGfoot, apg_fiducials) 
            mean, median, variance, skewness, kurt_values, std, entropy_pulse, center_freq, energy = self.statistical_measure_ppg(ppg_signal, PPGfoot, F_s, PPppg_med)
            class_waveform = self.PPG_wave_class(ppg_signal, ppg_fiducials)       
        else:
            AUCos_N, AUCso_N, T_os_f, T_so_f, T_so_cd, A_AC, A_sp, A_off, mean_slope_os,\
            mean_slope_so, PW_10, PW_25, PW_33, PW_50, PW_66, PW_75 = np.nan*np.ones(16)
            ACppg_med = np.nan
            systolic_areas, diastolic_areas, IPA, T_sys, T_dias, DN_exists, A_dn = np.nan*np.ones(7)
            AUCow_N, AUCwo_N, T_ow_f, T_wo_f = np.nan*np.ones(4)
            pat, dpat = np.nan*np.ones(2)
            PW_ms = np.nan
            pulse_areas = np.nan
            pulse_int_ms, pulse_rate = np.nan*np.ones(2)
            Pamp_diff_ppg = np.nan
            Delta_T_sd, RI, Delta_A_dn_dp, T_dn_dp_ms, SI = np.nan*np.ones(5)
            max_upslope = np.nan
            Aug_index = np.nan
            mean, median, variance, skewness, kurt_values, std, entropy_pulse, center_freq, energy = np.nan*np.ones(9)
            class_waveform = np.nan
        
  
        
        if signals_filt['resp'].sum() != 0:
            PPresp = np.diff(Resp_i)/F_s #in s
            resp_rate = 60/PPresp #in Bpm (Breaths per min.)
            inspiration_time = self.calculate_inspiration_time(Resp_i, Resp_e, F_s)
            exp_time = self.calculate_expiration_time(Resp_i, Resp_e, F_s)
            insp_exp_rat = self.calculate_insp_exp_ratio(Resp_i, Resp_e, F_s)
        elif signals_filt['edr'].sum() != 0:
            PPresp = np.diff(edr_i)/F_s #in s
            resp_rate = 60/PPresp #in Bpm (Breaths per min.)
            inspiration_time = self.calculate_inspiration_time(edr_i, edr_e, F_s)
            exp_time = self.calculate_expiration_time(edr_i, edr_e, F_s)
            insp_exp_rat = self.calculate_insp_exp_ratio(edr_i, edr_e, F_s)
        elif signals_filt['pdr'].sum() != 0:
            PPresp = np.diff(pdr_i)/F_s #in s
            resp_rate = 60/PPresp #in Bpm (Breaths per min.)
            inspiration_time = self.calculate_inspiration_time(pdr_i, pdr_e, F_s)
            exp_time = self.calculate_expiration_time(pdr_i, pdr_e, F_s)
            insp_exp_rat = self.calculate_insp_exp_ratio(pdr_i, pdr_e, F_s)
        else:
            PPresp, resp_rate, inspiration_time, exp_time, insp_exp_rat =  np.nan*np.ones(5)
            
  
        if signals_filt['abp'].sum() != 0: 
            abpHR, A_SBP, A_DBP, MAP, PP, CO, TPR, ShockIndex_abp  = self.ABP_features(abp_signal, SBP_final, DBP_final, F_s)
            
        else:
            abpHR, A_SBP, A_DBP, MAP, PP, CO, TPR, ShockIndex_abp = np.nan*np.ones(8)

        
        features   =    {'ECG':
                         {'T_rr_ms': RRecg,
                         'HR_bpm': (60*1000/RRecg), 
                         'edr_rate_Bpm': pd.Series(60/PPedr) #in s
                         },
                         
                         'HRV':
                         {'HRV_rqa_REC' : hrv_RecurrenceRate,
                         'HRV_rqa_DET' : hrv_Determinism,
                         'HRV_rqa_LAM' : hrv_Laminarity
                         },
                         
                         
                         'PPG':
                         {'AUC_pulse_nu':pd.Series(pulse_areas),
                         'AUC_sys_nu':pd.Series(systolic_areas),
                         'AUC_dias_nu':pd.Series(diastolic_areas),
                         'IPA':pd.Series(IPA),
                         'AUCow_nu':pd.Series(AUCow_N),
                         'AUCwo_nu':pd.Series(AUCwo_N), 
                         'AUCos_nu':pd.Series(AUCos_N),
                         'AUCso_nu':pd.Series(AUCso_N),
                         'Tsys_ms':pd.Series(T_sys),
                         'Tdias_ms':pd.Series(T_dias),
                         'T_ow_ms':pd.Series(T_ow_f),
                         'T_wo_ms':pd.Series(T_wo_f),
                         'T_os_ms':pd.Series(T_os_f),
                         'T_so_ms':pd.Series(T_so_f),
                         'T_so_cd_ms':pd.Series(T_so_cd),
                         'A_AC':pd.Series(A_AC),
                         'A_off':pd.Series(A_off),
                         'A_sp':pd.Series(A_sp),
                         'DN_exists': DN_exists,
                         'A_dn':pd.Series(A_dn),
                         'Ton_off_ms':pd.Series(PW_ms),
                         'Tsp_sp_ms':pd.Series(pulse_int_ms),
                         'PR':pd.Series(pulse_rate),
                         'mean_slope_os':pd.Series(mean_slope_os),
                         'mean_slope_so':pd.Series(mean_slope_so),
                         'Delta_T_sd':pd.Series(Delta_T_sd),
                         'RI':pd.Series(RI),
                         'SI':pd.Series(SI),
                         'Delta_A_dn_dp':pd.Series(Delta_A_dn_dp),
                         'T_dn_dp_ms':pd.Series(T_dn_dp_ms),
                         'ppg_class': class_waveform,
                         'max_upslope' : pd.Series(max_upslope),
                         'pat_ms' :pd.Series(pat), 
                         'dpat_ms': pd.Series(dpat),
                         'Delta_A_sp_sp': pd.Series(Pamp_diff_ppg),
                         'Augmentation_index' : pd.Series(Aug_index),
                         'center_freq_ppg_Hz': pd.Series(center_freq),
                         'mean_ppg' : pd.Series(mean), 
                         'median_ppg' : pd.Series(median),
                         'variance_ppg': pd.Series(variance), 
                         'skewness_ppg' : pd.Series(skewness), 
                         'kurtosis_ppg' : pd.Series(kurt_values), 
                         'std_ppg' : pd.Series(std), 
                         'entropy_ppg': pd.Series(entropy_pulse),
                         'energy_ppg':pd.Series(energy),
                         'PW_10':pd.Series(PW_10),
                         'PW_25':pd.Series(PW_25),
                         'PW_33':pd.Series(PW_33),
                         'PW_50':pd.Series(PW_50),
                         'PW_66':pd.Series(PW_66),
                         'PW_75':pd.Series(PW_75)
                         },
                         
                         
                         'RESP':    
                         {'insp_time' : pd.Series(inspiration_time), 
                         'exp_time' : pd.Series(exp_time), 
                         'insp_exp_ratio': pd.Series(insp_exp_rat), 
                         'resp_width_PPresp_s':pd.Series(PPresp), #in s,
                         'resp_rate_Bpm': pd.Series(resp_rate) #in Bpm
                         },
                         
                         
                         'ABP':
                         {'SBP' : pd.Series(A_SBP),
                         'DBP' : pd.Series(A_DBP),
                         'HR_ABP' : pd.Series(abpHR),
                         'MAP':pd.Series(MAP),
                         'PP_ABP':pd.Series(PP),
                         'CO':pd.Series(CO),
                         'TPR':pd.Series(TPR),
                         'ShockIndex': pd.Series(ShockIndex_abp)
                         }    
                         
                         }
        features['HRV'].update(hrv_dict)
        features['HRV'].update(MSE_dict)
        features['ECG'].update(self.ecgDeli_feat_dict)
        features['ECG'].update(self.ecgHB_feat_dict)
        features['PPG'].update(self.ppgHB_feat_dict)
        
        self.features = features
        return features
    

    def summary_statistics(self, stat_method='median'):
        """
        Compute summary statistics from nested self.features.

        Parameters:
            stat_method (str): One of 'mean', 'median', 'min', 'max', 'p25', 'p75', 'all'.
            'all' will compute all statistics in 16 dimensions

        Returns:
            flat_dict (dict): Flattened dictionary with computed stats
        """
        features = self.feature_extraction()  
        # Nested dict: e.g. {'ECG': {...}, 'PPG': {...}}
        
        # Flattening the nested dict
        flat_dict = {}
        for modality, modality_feats in features.items():
            for feat_name, feat_values in modality_feats.items():
                key = f"{feat_name}"
                # Exculuding single values (e.g., HRV features)
                if isinstance(feat_values, (list, np.ndarray, pd.Series)) and len(feat_values) > 1:  
                    stat_result = self.compute_stat(np.array(feat_values), stat_method)
                    for stat_k, stat_v in stat_result.items():
                        flat_dict[f"{key}_{stat_k}"] = stat_v
                elif isinstance(feat_values, (int, float, str, bool)):  # Handle single values
                    flat_dict[f"{key}"] = feat_values 
                elif feat_values is None or len(feat_values) == 0:  # Handle None or empty values
                    metr = ['mean', 'median', 'std', 'mad', 'iqr', 'min', 'max', 'p25', 'p75', 
                            'skewness', 'kurtosis', 'entropy', 'slope', 'autocorr_lag1', 'gini_index', 'valid_n']
                    for m in metr:
                        flat_dict[f"{key}_{m}"] = np.nan                
                else: pass
        return flat_dict

    



# --- After class is defined ---

# Automatically bind all functions from utils
custom_funcs = [
    (name, func)
    for name, func in inspect.getmembers(utils, inspect.isfunction)
    if inspect.getmodule(func) == utils
]

for name, func in custom_funcs:
    setattr(physiomarkers_from_biosignals, name, func) #staticmethod(func)


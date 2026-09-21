
"""
LLM-Based Sepsis Physiophenotype Interpretation Tool

This module provides comprehensive analysis and interpretation of sepsis physiophenotypes
using Large Language Models (OpenAI GPT or Claude) for clinical insight generation.

Author: Tilendra
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
from typing import Dict, List, Tuple, Optional
import openai
from anthropic import Anthropic
import time
import re

warnings.filterwarnings('ignore')

class SepsisPhysiophenotypeInterpreter:
    """
    A comprehensive tool for interpreting sepsis physiophenotypes using LLM analysis
    """
    
    def __init__(self, 
                 api_provider='openai',  # 'openai' or 'anthropic'
                 api_key: str = None,
                 model_name: str = None):
        """
        Initialize the interpreter with LLM configuration
        
        Args:
            api_provider: 'openai' or 'anthropic'
            api_key: API key for the chosen provider
            model_name: Model name (e.g., 'gpt-4o-mini', 'claude-3-sonnet-20240229')
        """
        self.api_provider = api_provider
        self.api_key = api_key
        
        if api_provider == 'openai':
            openai.api_key = api_key
            self.model_name = model_name or 'gpt-4o-mini'
            self.client = openai.OpenAI(api_key=api_key)
        elif api_provider == 'anthropic':
            self.model_name = model_name or 'claude-3-sonnet-20240229'
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError("api_provider must be 'openai' or 'anthropic'")
        
        self.feature_categories = self._define_feature_categories()
        self.phenotype_interpretations = {}
        
    
    def _define_feature_categories(self) -> Dict:
        """Define physiological feature categories for better interpretation"""
        return {
            # ECG-derived temporal and morphological features
            'ecg_temporal': [
                'ECG_T_rr_ms', 'ECG_HR_bpm', 'ECG_edr_rate_Bpm',
                'ECG_atrialSys_phase', 'ECG_ventSys_phase',
                'ECG_P_duration', 'ECG_QRS_duration', 'ECG_T_duration',
                'ECG_pr_interval_ms', 'ECG_pr_segment_ms',
                'ECG_st_interval_ms', 'ECG_st_segment_ms',
                'ECG_qt_interval_ms', 'ECG_qtc_interval_ms'
            ],
            'ecg_frequency': [
                'ECG_ecgBeatCenterFreq',
                'ECG_ecgInterbeat_eigval1', 'ECG_ecgInterbeat_eigval2', 'ECG_ecgInterbeat_eigval3'
            ],

            # HRV categories
            'hrv_time_domain': [
                'HRV_HRV_MeanNN', 'HRV_HRV_SDNN', 'HRV_HRV_SDANN1', 'HRV_HRV_SDNNI1',
                'HRV_HRV_SDANN2', 'HRV_HRV_SDNNI2', 'HRV_HRV_SDANN5', 'HRV_HRV_SDNNI5',
                'HRV_HRV_RMSSD', 'HRV_HRV_SDSD', 'HRV_HRV_pNN50', 'HRV_HRV_pNN20',
                'HRV_HRV_MinNN', 'HRV_HRV_MaxNN', 'HRV_HRV_MedianNN', 'HRV_HRV_MadNN'
            ],
            'hrv_frequency_domain': [
                'HRV_HRV_ULF', 'HRV_HRV_VLF', 'HRV_HRV_LF', 'HRV_HRV_HF', 'HRV_HRV_VHF',
                'HRV_HRV_TP', 'HRV_HRV_LFHF', 'HRV_HRV_LFn', 'HRV_HRV_HFn', 'HRV_HRV_LnHF'
            ],
            'hrv_nonlinear': [
                'HRV_HRV_SD1', 'HRV_HRV_SD2', 'HRV_HRV_SD1SD2',
                'HRV_HRV_ApEn', 'HRV_HRV_SampEn', 'HRV_HRV_ShanEn', 'HRV_HRV_FuzzyEn',
                'HRV_HRV_MSEn', 'HRV_HRV_CMSEn', 'HRV_HRV_RCMSEn',
                'HRV_HRV_CD', 'HRV_HRV_HFD', 'HRV_HRV_KFD', 'HRV_HRV_LZC'
            ],

            # PPG-derived
            'ppg_waveform': [
                'PPG_AUC_pulse_nu', 'PPG_AUC_sys_nu', 'PPG_AUC_dias_nu',
                'PPG_IPA', 'PPG_AUCow_nu', 'PPG_AUCwo_nu', 'PPG_AUCos_nu', 'PPG_AUCso_nu',
                'PPG_Tsys_ms', 'PPG_Tdias_ms', 'PPG_Ton_off_ms',
                'PPG_PR', 'PPG_RI', 'PPG_SI', 'PPG_Augmentation_index'
            ],
            'ppg_morphology': [
                'PPG_ppg_class', 'PPG_max_upslope', 'PPG_pat_ms', 'PPG_dpat_ms',
                'PPG_mean_slope_os', 'PPG_mean_slope_so', 'PPG_DN_exists'
            ],
            'ppg_statistics': [
                'PPG_mean_ppg', 'PPG_median_ppg', 'PPG_variance_ppg',
                'PPG_skewness_ppg', 'PPG_kurtosis_ppg', 'PPG_entropy_ppg', 'PPG_energy_ppg'
            ],
            'ppg_frequency': [
                'PPG_center_freq_ppg_Hz', 'PPG_ppgBeatCenterFreq',
                'PPG_ppgInterbeat_eigval1', 'PPG_ppgInterbeat_eigval2', 'PPG_ppgInterbeat_eigval3'
            ],

            # Respiration
            'respiratory': [
                'RESP_insp_time', 'RESP_exp_time', 'RESP_insp_exp_ratio',
                'RESP_resp_width_PPresp_s', 'RESP_resp_rate_Bpm'
            ],

            # Arterial Blood Pressure
            'abp': [
                'ABP_SBP', 'ABP_DBP', 'ABP_MAP', 'ABP_PP_ABP', 'ABP_HR_ABP',
                'ABP_CO', 'ABP_TPR', 'ABP_ShockIndex'
            ],

             # qSOFA Score
            'qSOFA': [
                'qSOFA_rr', 'qSOFA_sbp', 'qSOFA_gcs', 'qSOFA_total'
            ],
            
            # Statistical aggregations
            'summary_statistical_suffixes': [
                '_mean', '_p25', '_p75', '_skewness', '_entropy'
            ] 
        }
    
    def _get_feature_context(self, feature_name: str) -> str:
        """
        Get detailed clinical context for physiological features based on MPT feature reference
        """
        feature_contexts = {
            # ECG Features
            'T_rr_ms': 'RR interval (successive R-peaks interval) - reflects heart rhythm regularity',
            'HR_bpm': 'Heart Rate - primary cardiac vital sign',
            'edr_rate_Bpm': 'Respiratory rate estimated from ECG - cardiorespiratory coupling',
            'atrialSys_phase': 'Phase of atrial systole - atrial contraction timing',
            'ventSys_phase': 'Phase of ventricular systole - ventricular contraction timing',
            'P_duration': 'Duration of P wave - atrial depolarization time',
            'QRS_duration': 'Duration of QRS complex - ventricular depolarization time',
            'T_duration': 'Duration of T wave - ventricular repolarization time',
            'pr_interval_ms': 'PR interval - atrioventricular conduction time',
            'pr_segment_ms': 'PR segment - atrioventricular conduction segment',
            'st_interval_ms': 'ST interval - early ventricular repolarization',
            'st_segment_ms': 'ST segment - early ventricular repolarization segment',
            'qt_interval_ms': 'QT interval - total ventricular electrical activity',
            'qtc_interval_ms': 'Corrected QT interval - heart rate corrected ventricular repolarization',
            'ecgBeatCenterFreq': 'Center frequency of ECG beat - dominant frequency in ECG spectrum',
            'ecgInterbeat_eigval1': '1st eigenvalue of interbeat ECG covariance matrix - dominant interbeat ECG mode',
            'ecgInterbeat_eigval2': '2nd eigenvalue of interbeat ECG covariance matrix - secondary interbeat ECG mode',
            'ecgInterbeat_eigval3': '3rd eigenvalue of interbeat ECG covariance matrix - tertiary interbeat ECG mode',

            # HRV Features - RQA
            'rqa_REC': 'Recurrence rate from RQA of HRV - probability of repeated states (HRV pattern repetition)',
            'rqa_DET': 'Determinism from RQA of HRV - predictability of HRV patterns',
            'rqa_LAM': 'Laminarity from RQA of HRV - presence of laminar phases in HRV dynamics',

            # HRV Features - Time Domain
            'HRV_MeanNN': 'Mean of NN intervals - average heart rhythm',
            'HRV_SDNN': 'Standard deviation of NN intervals - overall HRV, autonomic balance',
            'HRV_SDANN1': 'SD of 1-min averages of NN intervals - long-term HRV variability',
            'HRV_SDNNI1': 'Mean of 1-min SD of NN intervals - short-term HRV variability',
            'HRV_SDANN2': 'SD of 2-min averages of NN intervals - intermediate-term HRV variability',
            'HRV_SDNNI2': 'Mean of 2-min SD of NN intervals - short-term HRV variability',
            'HRV_SDANN5': 'SD of 5-min averages of NN intervals - intermediate-term HRV variability',
            'HRV_SDNNI5': 'Mean of 5-min SD of NN intervals - short-term HRV variability',
            'HRV_RMSSD': 'Root mean square of successive differences - short-term HRV, parasympathetic activity',
            'HRV_SDSD': 'SD of successive differences - short-term HRV variability',
            'HRV_CVNN': 'Coefficient of variation of NN intervals - normalized HRV',
            'HRV_CVSD': 'Coefficient of variation of successive differences - normalized short-term HRV',
            'HRV_MedianNN': 'Median of NN intervals - median heart rhythm',
            'HRV_MadNN': 'Median absolute deviation of NN intervals - robust measure of HRV dispersion',
            'HRV_MCVNN': 'Median-based coefficient of variation - normalized HRV based on median',
            'HRV_IQRNN': 'Interquartile range of NN intervals - robust measure of HRV range',
            'HRV_SDRMSSD': 'Ratio of SDNN to RMSSD - balance between long and short-term variability',
            'HRV_Prc20NN': '20th percentile of NN intervals - lower range of heart rhythm',
            'HRV_Prc80NN': '80th percentile of NN intervals - upper range of heart rhythm',
            'HRV_pNN50': 'Proportion of NN intervals differing by >50ms - parasympathetic tone',
            'HRV_pNN20': 'Proportion of NN intervals differing by >20ms - short-term HRV',
            'HRV_MinNN': 'Minimum of NN intervals - shortest R-R interval',
            'HRV_MaxNN': 'Maximum of NN intervals - longest R-R interval',
            'HRV_HTI': 'HRV triangular index - geometric measure of HRV',
            'HRV_TINN': 'Triangular interpolation of NN histogram - HRV distribution width',

            # HRV Features - Frequency Domain
            'HRV_ULF': 'Power in ultra-low frequency band (<0.003 Hz) - long-term regulation',
            'HRV_VLF': 'Power in very low frequency band (0.003–0.04 Hz) - thermoregulation, slow mechanisms',
            'HRV_LF': 'Power in low frequency band (0.04–0.15 Hz) - sympathetic and parasympathetic activity',
            'HRV_HF': 'Power in high frequency band (0.15–0.4 Hz) - parasympathetic activity, respiratory sinus arrhythmia',
            'HRV_VHF': 'Power in very high frequency band (>0.4 Hz) - very rapid HRV components',
            'HRV_TP': 'Total power - overall autonomic activity',
            'HRV_LFHF': 'Ratio of LF to HF - sympathovagal balance',
            'HRV_LFn': 'Normalized low frequency power - % of LF in total power',
            'HRV_HFn': 'Normalized high frequency power - % of HF in total power',
            'HRV_LnHF': 'Natural log of HF power - log-transformed parasympathetic activity',

            # HRV Features - Poincaré Plot
            'HRV_SD1': 'Poincaré plot SD1 - short-term variability, parasympathetic activity',
            'HRV_SD2': 'Poincaré plot SD2 - long-term variability, sympathetic activity',
            'HRV_SD1SD2': 'Ratio SD1/SD2 - balance between short and long-term variability',
            'HRV_S': 'Area of Poincaré ellipse - overall HRV scatter area',
            'HRV_CSI': 'Cardiac Sympathetic Index - derived from Poincaré plot (SD2/SD1)',
            'HRV_CVI': 'Cardiac Vagal Index - derived from Poincaré plot (SD1^2)',
            'HRV_CSI_Modified': 'Modified Cardiac Sympathetic Index - alternative sympathovagal balance measure',
            'HRV_PIP': 'Phase-rectified signal average index - non-linear HRV pattern measure',
            'HRV_IALS': 'Integral of absolute value of successive differences - cumulative HRV change',
            'HRV_PSS': 'Power spectral slope - slope of HRV power spectrum',
            'HRV_PAS': 'Power spectral asymmetry - asymmetry of HRV power spectrum',
            'HRV_GI': 'Gini index of NN interval distribution - inequality measure of HRV distribution',
            'HRV_SI': 'Slope index - derived from Poincaré plot',
            'HRV_AI': 'Asymmetry index - derived from Poincaré plot',
            'HRV_PI': 'Phase index - derived from Poincaré plot',
            'HRV_C1d': 'Chaos-based parameter C1d - chaos measure using derivatives',
            'HRV_C1a': 'Chaos-based parameter C1a - chaos measure using amplitudes',
            'HRV_SD1d': 'Derivative-based SD1 - Poincaré short-term variability (derivative)',
            'HRV_SD1a': 'Amplitude-based SD1 - Poincaré short-term variability (amplitude)',
            'HRV_C2d': 'Chaos-based parameter C2d - chaos measure using derivatives',
            'HRV_C2a': 'Chaos-based parameter C2a - chaos measure using amplitudes',
            'HRV_SD2d': 'Derivative-based SD2 - Poincaré long-term variability (derivative)',
            'HRV_SD2a': 'Amplitude-based SD2 - Poincaré long-term variability (amplitude)',
            'HRV_Cd': 'Chaos index using derivatives - overall chaos measure',
            'HRV_Ca': 'Chaos index using amplitudes - overall chaos measure',
            'HRV_SDNNd': 'SDNN based on derivative transformation - HRV variability (derivative)',
            'HRV_SDNNa': 'SDNN based on amplitude transformation - HRV variability (amplitude)',

            # HRV Features - Fractal & Multifractal
            'HRV_DFA_alpha1': 'DFA scaling exponent (short-term) - fractal properties (4-16 beats)',
            'HRV_MFDFA_alpha1_Width': 'Width of multifractal spectrum (alpha1) - multifractal variability range (short-term)',
            'HRV_MFDFA_alpha1_Peak': 'Peak of multifractal spectrum (alpha1) - dominant multifractal scaling (short-term)',
            'HRV_MFDFA_alpha1_Mean': 'Mean of multifractal spectrum (alpha1) - average multifractal scaling (short-term)',
            'HRV_MFDFA_alpha1_Max': 'Max of multifractal spectrum (alpha1) - maximum multifractal scaling (short-term)',
            'HRV_MFDFA_alpha1_Delta': 'Difference (Max - Min) in multifractal alpha1 - multifractal width (short-term)',
            'HRV_MFDFA_alpha1_Asymmetry': 'Asymmetry of multifractal alpha1 - skewness of spectrum (short-term)',
            'HRV_MFDFA_alpha1_Fluctuation': 'Fluctuation index in multifractal alpha1 - multifractal fluctuation (short-term)',
            'HRV_MFDFA_alpha1_Increment': 'Increment of multifractal alpha1 - multifractal change (short-term)',
            'HRV_DFA_alpha2': 'DFA scaling exponent (long-term) - fractal properties (16+ beats)',
            'HRV_MFDFA_alpha2_Width': 'Width of multifractal spectrum (alpha2) - multifractal variability range (long-term)',
            'HRV_MFDFA_alpha2_Peak': 'Peak of multifractal spectrum (alpha2) - dominant multifractal scaling (long-term)',
            'HRV_MFDFA_alpha2_Mean': 'Mean of multifractal spectrum (alpha2) - average multifractal scaling (long-term)',
            'HRV_MFDFA_alpha2_Max': 'Max of multifractal spectrum (alpha2) - maximum multifractal scaling (long-term)',
            'HRV_MFDFA_alpha2_Delta': 'Delta (Max - Min) of alpha2 - multifractal width (long-term)',
            'HRV_MFDFA_alpha2_Asymmetry': 'Asymmetry of multifractal alpha2 - skewness of spectrum (long-term)',
            'HRV_MFDFA_alpha2_Fluctuation': 'Fluctuation in alpha2 - multifractal fluctuation (long-term)',
            'HRV_MFDFA_alpha2_Increment': 'Increment in alpha2 - multifractal change (long-term)',

            # HRV Features - Nonlinear
            'HRV_ApEn': 'Approximate Entropy - signal regularity and complexity',
            'HRV_SampEn': 'Sample Entropy - pattern regularity in heart rhythm',
            'HRV_ShanEn': 'Shannon Entropy - measure of HRV signal complexity',
            'HRV_FuzzyEn': 'Fuzzy Entropy - measure of HRV signal complexity using fuzzy sets',
            'HRV_MSEn': 'Multiscale Entropy (mean of scales) - complexity across multiple time scales',
            'HRV_CMSEn': 'Composite Multiscale Entropy - refined multiscale complexity measure',
            'HRV_RCMSEn': 'Refined Composite Multiscale Entropy - advanced multiscale complexity measure',
            'HRV_CD': 'Correlation Dimension - measure of HRV attractor complexity',
            'HRV_HFD': 'Higuchi Fractal Dimension - measure of HRV signal irregularity',
            'HRV_KFD': 'Katz Fractal Dimension - measure of HRV signal complexity',
            'HRV_LZC': 'Lempel-Ziv Complexity - measure of HRV signal pattern complexity',
            'HRV_MSE_0': 'Multiscale Entropy at scale 0 - complexity at finest scale',
            'HRV_MSE_1': 'Multiscale Entropy at scale 1 - complexity at scale 1',
            'HRV_MSE_2': 'Multiscale Entropy at scale 2 - complexity at scale 2',
            'HRV_MSE_3': 'Multiscale Entropy at scale 3 - complexity at scale 3',
            'HRV_MSE_4': 'Multiscale Entropy at scale 4 - complexity at scale 4',
            'HRV_MSE_5': 'Multiscale Entropy at scale 5 - complexity at scale 5',
            'HRV_MSE_6': 'Multiscale Entropy at scale 6 - complexity at scale 6',
            'HRV_MSE_7': 'Multiscale Entropy at scale 7 - complexity at scale 7',
            'HRV_MSE_8': 'Multiscale Entropy at scale 8 - complexity at scale 8',
            'HRV_MSE_9': 'Multiscale Entropy at scale 9 - complexity at scale 9',
            'HRV_MSE_10': 'Multiscale Entropy at scale 10 - complexity at scale 10',
            'HRV_MSE_11': 'Multiscale Entropy at scale 11 - complexity at scale 11',
            'HRV_MSE_12': 'Multiscale Entropy at scale 12 - complexity at scale 12',
            'HRV_MSE_13': 'Multiscale Entropy at scale 13 - complexity at scale 13',
            'HRV_MSE_14': 'Multiscale Entropy at scale 14 - complexity at scale 14',
            'HRV_MSE_15': 'Multiscale Entropy at scale 15 - complexity at scale 15',
            'HRV_MSE_16': 'Multiscale Entropy at scale 16 - complexity at scale 16',
            'HRV_MSE_17': 'Multiscale Entropy at scale 17 - complexity at scale 17',
            'HRV_MSE_18': 'Multiscale Entropy at scale 18 - complexity at scale 18',
            'HRV_MSE_19': 'Multiscale Entropy at scale 19 - complexity at scale 19',

            # PPG Features - Waveform
            'PPG_AUC_pulse_nu': 'Area under pulse wave (normalized) - stroke volume surrogate',
            'PPG_AUC_sys_nu': 'Area under systolic part (normalized) - systolic blood flow',
            'PPG_AUC_dias_nu': 'Area under diastolic part (normalized) - diastolic filling',
            'PPG_IPA': 'Inflection point area - arterial stiffness indicator',
            'PPG_AUCow_nu': 'AUC onset to wave peak (normalized) - early systolic area',
            'PPG_AUCwo_nu': 'AUC wave peak to offset (normalized) - late systolic + diastolic area',
            'PPG_AUCos_nu': 'AUC onset to shoulder (normalized) - early systolic + shoulder area',
            'PPG_AUCso_nu': 'AUC shoulder to offset (normalized) - shoulder to diastolic area',
            'PPG_Tsys_ms': 'Duration of systolic phase - left ventricular ejection time',
            'PPG_Tdias_ms': 'Duration of diastolic phase - filling time',
            'PPG_T_ow_ms': 'Onset to wave peak time - systolic rise time',
            'PPG_T_wo_ms': 'Wave peak to offset time - systolic + early diastolic time',
            'PPG_T_os_ms': 'Onset to shoulder time - systolic + shoulder time',
            'PPG_T_so_ms': 'Shoulder to offset time - shoulder to early diastolic time',
            'PPG_T_so_cd_ms': 'Shoulder to dicrotic notch (or centroid) - dicrotic notch timing',
            'PPG_A_AC': 'Amplitude of AC (pulsatile) component - pulse strength',
            'PPG_A_off': 'Amplitude at offset - pulse baseline',
            'PPG_A_sp': 'Amplitude at shoulder peak - shoulder amplitude',
            'PPG_DN_exists': 'Dicrotic notch exists (binary) - presence of dicrotic notch',
            'PPG_A_dn': 'Amplitude at dicrotic notch - dicrotic notch amplitude',
            'PPG_Ton_off_ms': 'Onset to offset time - total pulse duration',
            'PPG_Tsp_sp_ms': 'Shoulder peak to shoulder peak interval - pulse rate indicator',
            'PPG_PR': 'Pulse rate from PPG - peripheral heart rate',
            'PPG_mean_slope_os': 'Mean slope from onset to shoulder - early systolic upstroke',
            'PPG_mean_slope_so': 'Mean slope from shoulder to offset - shoulder to diastolic decay',
            'PPG_Delta_T_sd': 'Time difference between systolic and diastolic peak - peak timing diff',
            'PPG_RI': 'Reflection Index - arterial stiffness and wave reflection',
            'PPG_SI': 'Stiffness Index - large artery stiffness',
            'PPG_Delta_A_dn_dp': 'Amplitude diff between dicrotic notch & diastolic peak - dicrotic amp diff',
            'PPG_T_dn_dp_ms': 'Time diff between dicrotic notch & diastolic peak - dicrotic time diff',
            'PPG_ppg_class': 'Pulse wave morphology class - categorical shape descriptor',
            'PPG_max_upslope': 'Maximum slope in rising edge - systolic upstroke steepness',
            'PPG_pat_ms': 'Pulse arrival time - arterial stiffness and blood pressure',
            'PPG_dpat_ms': 'Derivative of PAT over time - change in pulse arrival',
            'PPG_Delta_A_sp_sp': 'Amplitude difference between shoulder peaks - shoulder amp variability',
            'PPG_Augmentation_index': 'Measure of arterial stiffness - pressure wave augmentation',
            'PPG_center_freq_ppg_Hz': 'Center frequency of PPG spectrum - dominant frequency in PPG',
            'PPG_mean_ppg': 'Mean of raw PPG signal - average PPG level',
            'PPG_median_ppg': 'Median of raw PPG signal - median PPG level',
            'PPG_variance_ppg': 'Variance of PPG - PPG signal variability',
            'PPG_skewness_ppg': 'Skewness of PPG distribution - PPG signal asymmetry',
            'PPG_kurtosis_ppg': 'Kurtosis of PPG distribution - PPG signal peakedness',
            'PPG_std_ppg': 'Standard deviation of PPG - PPG signal dispersion',
            'PPG_entropy_ppg': 'Entropy of PPG - PPG signal complexity',
            'PPG_energy_ppg': 'Energy of PPG signal - total power of PPG',
            'PPG_PW_10': 'Pulse width at 10% amplitude - pulse width measure',
            'PPG_PW_25': 'Pulse width at 25% amplitude - pulse width measure',
            'PPG_PW_33': 'Pulse width at 33% amplitude - pulse width measure',
            'PPG_PW_50': 'Pulse width at 50% amplitude - pulse width measure (FWHM)',
            'PPG_PW_66': 'Pulse width at 66% amplitude - pulse width measure',
            'PPG_PW_75': 'Pulse width at 75% amplitude - pulse width measure',
            'PPG_ppgBeatCenterFreq': 'Dominant frequency in PPG beats - dominant frequency in PPG spectrum',
            'PPG_ppgInterbeat_eigval1': '1st eigenvalue of inter-beat PPG matrix - dominant interbeat PPG mode',
            'PPG_ppgInterbeat_eigval2': '2nd eigenvalue of inter-beat PPG matrix - secondary interbeat PPG mode',
            'PPG_ppgInterbeat_eigval3': '3rd eigenvalue of inter-beat PPG matrix - tertiary interbeat PPG mode',

            # Respiratory Features
            'RESP_insp_time': 'Inspiration time - respiratory muscle function',
            'RESP_exp_time': 'Expiration time - respiratory mechanics',
            'RESP_insp_exp_ratio': 'I:E ratio - breathing pattern efficiency',
            'RESP_resp_width_PPresp_s': 'Width of P-P respiratory cycle - respiratory period',
            'RESP_resp_rate_Bpm': 'Respiratory rate - ventilatory status',

             # qSOFA Features
            'qSOFA_rr': 'qSOFA respiratory rate score - respiratory dysfunction (score 1 if RR >= 22)',   
            'qSOFA_sbp': 'qSOFA systolic blood pressure score - hemodynamic instability (score 1 if SBP <= 100 mmHg)',
            'qSOFA_gcs': 'qSOFA Glasgow Coma Scale score - neurologic dysfunction (score 1 if GCS < 15)',
            'qSOFA_total': 'Total qSOFA score (0-3) - sepsis severity indicator (>=2 suggests higher risk)'
        }

        # Look for matches in feature name (handles prefixes like ECG_, HRV_, PPG_, RESP_ and suffixes like _mean, _std, etc.)
        for key, context in feature_contexts.items():
            # Check if the base key (without prefixes/suffixes) is in the feature name
            if key.lower() in feature_name.lower():
                # If found, return the base context
                return context

        # Handle statistical suffixes if the base feature was found via the loop above
        # This part runs *after* the loop to ensure base feature context is prioritized
        # but also handles cases where the suffix is the only part matching
        # Check for specific suffixes and remove them to find the base feature
        base_feature = feature_name
        suffix_added_context = ""
        if feature_name.lower().endswith('_mean'):
            base_feature = feature_name[:-5] # Remove '_mean'
            suffix_added_context = " (average value)"
        elif feature_name.lower().endswith('_p25'):
            base_feature = feature_name[:-4] # Remove '_p25'
            suffix_added_context = " (25th percentile - lower range)"
        elif feature_name.lower().endswith('_p75'):
            base_feature = feature_name[:-4] # Remove '_p75'
            suffix_added_context = " (75th percentile - upper range)"
        elif feature_name.lower().endswith('_skewness'):
            base_feature = feature_name[:-9] # Remove '_skewness'
            suffix_added_context = " (distribution asymmetry)"
        elif feature_name.lower().endswith('_entropy'):
            base_feature = feature_name[:-8] # Remove '_entropy'
            suffix_added_context = " (signal complexity/irregularity)"
        # Add more suffixes if needed (e.g., _std, _variance, _median, etc.)
        # elif feature_name.lower().endswith('_std'):
        #     base_feature = feature_name[:-4]
        #     suffix_added_context = " (variability measure)"
        # elif feature_name.lower().endswith('_median'):
        #     base_feature = feature_name[:-7]
        #     suffix_added_context = " (median value)"

        if suffix_added_context: # If a suffix was found and removed
            base_context = self._get_feature_context(base_feature) # Recursively call to get base context
            if base_context != "Feature context not available":
                return f"{base_context}{suffix_added_context}" # Append suffix meaning to base meaning
            # If base context was not found, return the default below

        return "Feature context not available"

    
    
#     def _get_feature_context(self, feature_name: str) -> str:
#         """
#         Get detailed clinical context for physiological features based on MPT feature reference
#         """
#         feature_contexts = {
#             # ECG Features
#             'T_rr_ms': 'RR interval (successive R-peaks interval) - reflects heart rhythm regularity',
#             'HR_bpm': 'Heart Rate - primary cardiac vital sign',
#             'edr_rate_Bpm': 'Respiratory rate estimated from ECG - cardiorespiratory coupling',
#             'atrialSys_phase': 'Phase of atrial systole - atrial contraction timing',
#             'ventSys_phase': 'Phase of ventricular systole - ventricular contraction timing',
#             'P_duration': 'Duration of P wave - atrial depolarization time',
#             'QRS_duration': 'Duration of QRS complex - ventricular depolarization time',
#             'T_duration': 'Duration of T wave - ventricular repolarization time',
#             'pr_interval_ms': 'PR interval - atrioventricular conduction time',
#             'st_interval_ms': 'ST interval - early ventricular repolarization',
#             'qt_interval_ms': 'QT interval - total ventricular electrical activity',
#             'qtc_interval_ms': 'Corrected QT interval - heart rate corrected ventricular repolarization',
            
#             # HRV Features - Time Domain
#             'MeanNN': 'Mean of NN intervals - average heart rhythm',
#             'SDNN': 'Standard deviation of NN intervals - overall HRV, autonomic balance',
#             'RMSSD': 'Root mean square of successive differences - short-term HRV, parasympathetic activity',
#             'pNN50': 'Proportion of NN intervals differing by >50ms - parasympathetic tone',
#             'CVNN': 'Coefficient of variation of NN intervals - normalized HRV',
#             'HTI': 'HRV triangular index - geometric measure of HRV',
#             'TINN': 'Triangular interpolation of NN histogram - HRV distribution width',
            
#             # HRV Features - Frequency Domain
#             'ULF': 'Ultra-low frequency power (<0.003 Hz) - long-term regulation',
#             'VLF': 'Very low frequency power (0.003–0.04 Hz) - thermoregulation, slow mechanisms',
#             'LF': 'Low frequency power (0.04–0.15 Hz) - sympathetic and parasympathetic activity',
#             'HF': 'High frequency power (0.15–0.4 Hz) - parasympathetic activity, respiratory sinus arrhythmia',
#             'LFHF': 'LF/HF ratio - sympathovagal balance',
#             'TP': 'Total power - overall autonomic activity',
            
#             # HRV Features - Nonlinear
#             'SD1': 'Poincaré plot SD1 - short-term variability, parasympathetic activity',
#             'SD2': 'Poincaré plot SD2 - long-term variability, sympathetic activity',
#             'ApEn': 'Approximate Entropy - signal regularity and complexity',
#             'SampEn': 'Sample Entropy - pattern regularity in heart rhythm',
#             'DFA_alpha1': 'DFA scaling exponent (short-term) - fractal properties',
#             'DFA_alpha2': 'DFA scaling exponent (long-term) - long-range correlations',
            
#             # PPG Features
#             'AUC_pulse': 'Area under pulse wave - stroke volume surrogate',
#             'AUC_sys': 'Area under systolic part - systolic blood flow',
#             'AUC_dias': 'Area under diastolic part - diastolic filling',
#             'IPA': 'Inflection point area - arterial stiffness indicator',
#             'Tsys_ms': 'Duration of systolic phase - left ventricular ejection time',
#             'Tdias_ms': 'Duration of diastolic phase - filling time',
#             'A_AC': 'AC component amplitude - pulse strength',
#             'PR': 'Pulse rate from PPG - peripheral heart rate',
#             'RI': 'Reflection Index - arterial stiffness and wave reflection',
#             'SI': 'Stiffness Index - large artery stiffness',
#             'Augmentation_index': 'Arterial stiffness measure - pressure wave augmentation',
#             'pat_ms': 'Pulse arrival time - arterial stiffness and blood pressure',
            
#             # Respiratory Features
#             'insp_time': 'Inspiration time - respiratory muscle function',
#             'exp_time': 'Expiration time - respiratory mechanics',
#             'insp_exp_ratio': 'I:E ratio - breathing pattern efficiency',
#             'resp_rate_Bpm': 'Respiratory rate - ventilatory status',
            
#             # ABP Features
#             'SBP': 'Systolic Blood Pressure - peak arterial pressure',
#             'DBP': 'Diastolic Blood Pressure - minimum arterial pressure',
#             'MAP': 'Mean Arterial Pressure - average perfusion pressure',
#             'PP_ABP': 'Pulse Pressure - arterial compliance indicator',
#             'CO': 'Cardiac Output - heart pumping efficiency',
#             'TPR': 'Total Peripheral Resistance - vascular resistance',
#             'ShockIndex': 'Heart Rate/SBP ratio - hemodynamic compromise indicator',
            
#              # qSOFA Features
#             'qSOFA_rr': 'qSOFA respiratory rate score - respiratory dysfunction',   
#             'qSOFA_sbp': 'qSOFA systolic blood pressure score - hemodynamic instability',
#             'qSOFA_gcs': 'qSOFA Glasgow Coma Scale score - neurologic dysfunction',
#             'qSOFA_total': 'Total qSOFA score - sepsis severity indicator'
#         }
        
#         # Look for matches in feature name (handles suffixes like _mean, _std, etc.)
#         for key, context in feature_contexts.items():
#             if key.lower() in feature_name.lower():
#                 return context
        
#         # Handle statistical suffixes
#         if '_mean' in feature_name:
#             base_feature = feature_name.replace('_mean', '')
#             base_context = self._get_feature_context(base_feature)
#             if base_context != "Feature context not available":
#                 return f"{base_context} (average value)"
#         elif '_std' in feature_name:
#             base_feature = feature_name.replace('_std', '')
#             base_context = self._get_feature_context(base_feature)
#             if base_context != "Feature context not available":
#                 return f"{base_context} (variability measure)"
#         elif '_p25' in feature_name:
#             base_feature = feature_name.replace('_p25', '')
#             base_context = self._get_feature_context(base_feature)
#             if base_context != "Feature context not available":
#                 return f"{base_context} (25th percentile - lower range)"
#         elif '_p75' in feature_name:
#             base_feature = feature_name.replace('_p75', '')
#             base_context = self._get_feature_context(base_feature)
#             if base_context != "Feature context not available":
#                 return f"{base_context} (75th percentile - upper range)"
#         elif '_skewness' in feature_name:
#             base_feature = feature_name.replace('_skewness', '')
#             base_context = self._get_feature_context(base_feature)
#             if base_context != "Feature context not available":
#                 return f"{base_context} (distribution asymmetry)"
#         elif '_entropy' in feature_name:
#             base_feature = feature_name.replace('_entropy', '')
#             base_context = self._get_feature_context(base_feature)
#             if base_context != "Feature context not available":
#                 return f"{base_context} (signal complexity/irregularity)"
        
#         return "Feature context not available"
    
    
    def analyze_cluster_characteristics(self,
                                     data: pd.DataFrame,
                                     cluster_labels: np.ndarray,
                                     feature_columns: List[str],
                                     shap_values: np.ndarray = None,
                                     feature_indices: Dict = None) -> Dict:
        """
        Analyze characteristics of each cluster/phenotype using SHAP values
        """
        print("Analyzing cluster characteristics using SHAP values...")

        cluster_analysis = {}
        unique_clusters = np.unique(cluster_labels)

        # Extract feature data
        feature_data = data[feature_columns]

        # Calculate overall mean once for efficiency
        overall_means = feature_data.mean().to_dict()

        # Debug SHAP structure
        if shap_values is not None:
            print(f"🔍 SHAP values shape: {shap_values.shape}")
            print(f"🔍 Cluster labels shape: {cluster_labels.shape}")
            print(f"🔍 Number of unique clusters: {len(unique_clusters)}")

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = feature_data[cluster_mask]

            # Calculate cluster means
            cluster_means = cluster_data.mean().to_dict()

            # Create a new dictionary to store directionality for each feature
            feature_directions = {}
            for feature in feature_columns:
                if feature in cluster_means and feature in overall_means:
                    # Compare cluster mean to overall mean
                    if cluster_means[feature] > overall_means[feature]:
                        feature_directions[feature] = "↑ Higher"
                    elif cluster_means[feature] < overall_means[feature]:
                        feature_directions[feature] = "↓ Lower"
                    else:
                        feature_directions[feature] = "≈ Normal"  # Or "No difference"
                else:
                    # Handle case where feature might be missing (unlikely here, but good practice)
                    feature_directions[feature] = "≈ Normal"

            # Basic statistics
            cluster_stats = {
                'n_patients': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(cluster_labels) * 100),
                'feature_means': cluster_means, # Use the calculated cluster_means
                'feature_stds': cluster_data.std().to_dict(),
            }

            # Use SHAP values for feature importance
            if shap_values is not None and feature_indices is not None:
                try:
                    # For 3D SHAP array (samples × features × classes)
                    # We need to get SHAP values for this specific cluster/class
                    if shap_values.ndim == 3:
                        # Get the class index (cluster_id should match class index in SHAP)
                        class_idx = int(cluster_id)

                        # Get SHAP values for this specific class across all samples
                        class_shap_values = shap_values[:, :, class_idx]  # Shape: (200, 273)

                        # Get SHAP values only for samples in this cluster
                        cluster_shap = class_shap_values[cluster_mask]  # Shape: (n_cluster_samples, 273)

                        print(f"🔍 Cluster {cluster_id}: {cluster_shap.shape[0]} samples, SHAP shape: {cluster_shap.shape}")

                    else:
                        # Handle 2D SHAP array (fallback)
                        cluster_shap = shap_values[cluster_mask]

                    # Calculate mean absolute SHAP values for feature importance
                    mean_abs_shap = np.mean(np.abs(cluster_shap), axis=0)

                    # Create feature importance mapping
                    feature_importance = {}
                    for feature in feature_columns:
                        if feature in feature_indices:
                            idx = feature_indices[feature]
                            feature_importance[feature] = float(mean_abs_shap[idx])

                    # Get top features based on SHAP importance
                    top_features_df = pd.DataFrame({
                        'feature': list(feature_importance.keys()),
                        'importance': list(feature_importance.values())
                    }).sort_values('importance', ascending=False)

                    # Create the cluster analysis entry, including feature_directions
                    cluster_analysis[f'Cluster_{cluster_id}'] = {
                        **cluster_stats,
                        'feature_importance': feature_importance,
                        'top_discriminative_features': top_features_df.head(10).to_dict('records'),
                        'shap_values': cluster_shap,
                        'feature_directions': feature_directions # Add the directions here
                    }


                except Exception as e:
                    print(f"⚠️  Error processing SHAP values for cluster {cluster_id}: {e}")
                    print("   Falling back to statistical analysis...")
                    top_features = self._fallback_statistical_analysis(
                        cluster_data, feature_data, cluster_mask, feature_columns)
                    # Create the cluster analysis entry for the fallback, including feature_directions
                    cluster_analysis[f'Cluster_{cluster_id}'] = {
                        **cluster_stats,
                        'top_discriminative_features': top_features,
                        'feature_directions': feature_directions # Add the directions here
                    }
            else:
                # Fallback to statistical analysis if SHAP not available
                print(f"Warning: SHAP values not provided for cluster {cluster_id}, using statistical fallback")
                top_features = self._fallback_statistical_analysis(
                    cluster_data, feature_data, cluster_mask, feature_columns)
                # Create the cluster analysis entry for the fallback, including feature_directions
                cluster_analysis[f'Cluster_{cluster_id}'] = {
                    **cluster_stats,
                    'top_discriminative_features': top_features,
                    'feature_directions': feature_directions # Add the directions here
                }

        return cluster_analysis



#     def analyze_cluster_characteristics(self, 
#                                      data: pd.DataFrame, 
#                                      cluster_labels: np.ndarray,
#                                      feature_columns: List[str],
#                                      shap_values: np.ndarray = None,
#                                      feature_indices: Dict = None) -> Dict:
#         """
#         Analyze characteristics of each cluster/phenotype using SHAP values
#         """
#         print("Analyzing cluster characteristics using SHAP values...")

#         cluster_analysis = {}
#         unique_clusters = np.unique(cluster_labels)

#         # Extract feature data
#         feature_data = data[feature_columns]

#         # Debug SHAP structure
#         if shap_values is not None:
#             print(f"🔍 SHAP values shape: {shap_values.shape}")
#             print(f"🔍 Cluster labels shape: {cluster_labels.shape}")
#             print(f"🔍 Number of unique clusters: {len(unique_clusters)}")

#         for cluster_id in unique_clusters:
#             cluster_mask = cluster_labels == cluster_id
#             cluster_data = feature_data[cluster_mask]
            

#             # Basic statistics
#             cluster_stats = {
#                 'n_patients': int(np.sum(cluster_mask)),
#                 'percentage': float(np.sum(cluster_mask) / len(cluster_labels) * 100),
#                 'feature_means': cluster_data.mean().to_dict(),
#                 'feature_stds': cluster_data.std().to_dict(),
#             }

#             # Use SHAP values for feature importance
#             if shap_values is not None and feature_indices is not None:
#                 try:
#                     # For 3D SHAP array (samples × features × classes)
#                     # We need to get SHAP values for this specific cluster/class
#                     if shap_values.ndim == 3:
#                         # Get the class index (cluster_id should match class index in SHAP)
#                         class_idx = int(cluster_id)

#                         # Get SHAP values for this specific class across all samples
#                         class_shap_values = shap_values[:, :, class_idx]  # Shape: (200, 273)

#                         # Get SHAP values only for samples in this cluster
#                         cluster_shap = class_shap_values[cluster_mask]  # Shape: (n_cluster_samples, 273)

#                         print(f"🔍 Cluster {cluster_id}: {cluster_shap.shape[0]} samples, SHAP shape: {cluster_shap.shape}")

#                     else:
#                         # Handle 2D SHAP array (fallback)
#                         cluster_shap = shap_values[cluster_mask]

#                     # Calculate mean absolute SHAP values for feature importance
#                     mean_abs_shap = np.mean(np.abs(cluster_shap), axis=0)

#                     # Create feature importance mapping
#                     feature_importance = {}
#                     for feature in feature_columns:
#                         if feature in feature_indices:
#                             idx = feature_indices[feature]
#                             feature_importance[feature] = float(mean_abs_shap[idx])

#                     # Get top features based on SHAP importance
#                     top_features_df = pd.DataFrame({
#                         'feature': list(feature_importance.keys()),
#                         'importance': list(feature_importance.values())
#                     }).sort_values('importance', ascending=False)

#                     cluster_analysis[f'Cluster_{cluster_id}'] = {
#                         **cluster_stats,
#                         'feature_importance': feature_importance,
#                         'top_discriminative_features': top_features_df.head(10).to_dict('records'),
#                         'shap_values': cluster_shap
#                     }

#                 except Exception as e:
#                     print(f"⚠️  Error processing SHAP values for cluster {cluster_id}: {e}")
#                     print("   Falling back to statistical analysis...")
#                     top_features = self._fallback_statistical_analysis(
#                         cluster_data, feature_data, cluster_mask, feature_columns)
#                     cluster_analysis[f'Cluster_{cluster_id}'] = {
#                         **cluster_stats,
#                         'top_discriminative_features': top_features
#                     }
#             else:
#                 # Fallback to statistical analysis if SHAP not available
#                 print(f"Warning: SHAP values not provided for cluster {cluster_id}, using statistical fallback")
#                 top_features = self._fallback_statistical_analysis(
#                     cluster_data, feature_data, cluster_mask, feature_columns)
#                 cluster_analysis[f'Cluster_{cluster_id}'] = {
#                     **cluster_stats,
#                     'top_discriminative_features': top_features
#                 }

#         return cluster_analysis

    def _get_top_discriminative_features_fallback(self, 
                                                significant_features: Dict, 
                                                effect_sizes: Dict, 
                                                n_top: int = 10) -> List[Dict]:
        """Fallback method for top features when SHAP is not available"""
        combined_scores = {}
        
        for feature in significant_features.keys():
            if feature in effect_sizes:
                # Combined score: inverse p-value * absolute effect size
                p_val = significant_features[feature]
                effect = abs(effect_sizes[feature])
                combined_scores[feature] = (1 / (p_val + 1e-10)) * effect
        
        # Sort by combined score
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_features = []
        for feature, score in sorted_features[:n_top]:
            top_features.append({
                'feature': feature,
                'p_value': significant_features[feature],
                'effect_size': effect_sizes[feature],
                'combined_score': score,
                'importance': score  # For compatibility with SHAP-based method
            })
        
        return top_features
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Make API call to chosen LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                if self.api_provider == 'openai':
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert clinical data scientist specializing in sepsis pathophysiology and critical care medicine."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1500,
                        temperature=0 #0.1
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.api_provider == 'anthropic':
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=1500,
                        temperature=0, #0.1,
                        system="You are an expert clinical data scientist specializing in sepsis pathophysiology and critical care medicine.",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text.strip()
                    
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

#     def _create_enhanced_feature_summary(self, top_features: List[Dict], shap_values: np.ndarray = None, feature_indices: Dict = None) -> str:
#         """Create feature summary with clinical context using SHAP-based importance"""
#         summary_lines = []
        
#         for i, feature_data in enumerate(top_features[:10], 1):
#             feature = feature_data['feature']
#             importance = feature_data.get('importance', 0)
            
#             # Determine directionality from SHAP if available
#             direction = "↑ Higher"  # Default
#             if shap_values is not None and feature_indices is not None and feature in feature_indices:
#                 feature_idx = feature_indices[feature]
#                 mean_shap = np.mean(shap_values[:, feature_idx])
#                 direction = "↑ Higher" if mean_shap > 0 else "↓ Lower"
            
#             # Categorize importance
#             max_importance = max([f.get('importance', 0) for f in top_features]) if top_features else 1
#             if max_importance > 0:
#                 relative_importance = importance / max_importance
#                 if relative_importance >= 0.66:
#                     importance_label = "High"
#                 elif relative_importance >= 0.33:
#                     importance_label = "Moderate"
#                 else:
#                     importance_label = "Low"
#             else:
#                 importance_label = "Medium"
            
#             # Get clinical context
#             context = self._get_feature_context(feature)
            
#             summary_lines.append(
#                 f"{i}. {feature}: {direction} values ({importance_label} importance, SHAP={importance:.4f})")
#             summary_lines.append(f"   Clinical meaning: {context}")
#             summary_lines.append("")  # Empty line for readability
        
#         return "\n".join(summary_lines)

    def _create_enhanced_feature_summary(self, top_features: List[Dict], shap_values: np.ndarray = None, feature_indices: Dict = None, feature_directions: Dict = None) -> str:
        """Create feature summary with clinical context using SHAP-based importance"""
        summary_lines = []
        for i, feature_data in enumerate(top_features[:10], 1):
            feature = feature_data['feature']
            importance = feature_data.get('importance', 0)

            # Get directionality from the pre-calculated dictionary
            direction = "≈ Normal"  # Default if not found
            if feature_directions is not None:
                direction = feature_directions.get(feature, "≈ Normal")

            # Categorize importance (same as before)
            max_importance = max([f.get('importance', 0) for f in top_features]) if top_features else 1
            if max_importance > 0:
                relative_importance = importance / max_importance
                if relative_importance >= 0.66:
                    magnitude = "High"
                elif relative_importance >= 0.33:
                    magnitude = "Moderate"
                else:
                    magnitude = "Low"
            else:
                magnitude = "Medium"

            # Get clinical context
            context = self._get_feature_context(feature)
            summary_lines.append(
                f"{i}. {feature}: {direction} values ({magnitude} importance, SHAP={importance:.4f}). Clinical meaning: {context}")
            summary_lines.append("")  # Empty line for readability
            print(f'feature_data: {feature_data}')
            print(f'summary_lines: {summary_lines}')

        return "\n".join(summary_lines)

#     def _create_enhanced_feature_summary(self, top_features: List[Dict], shap_values: np.ndarray = None, feature_indices: Dict = None) -> str:
#         """Create feature summary with clinical context using SHAP-based importance"""
#         summary_lines = []

#         for i, feature_data in enumerate(top_features[:10], 1):
#             feature = feature_data['feature']
#             importance = feature_data.get('importance', 0)

#             # Determine directionality from SHAP if available
#             direction = "↑ Higher"  # Default
#             magnitude = "Medium"  # Default

#             if shap_values is not None and feature_indices is not None and feature in feature_indices:
#                 try:
#                     feature_idx = feature_indices[feature]

#                     # For 2D SHAP array (samples × features)
#                     if shap_values.ndim == 2:
#                         mean_shap = np.mean(shap_values[:, feature_idx])
#                     # For 3D SHAP array (samples × features × classes) - we're already passing cluster-specific 2D SHAP
#                     else:
#                         mean_shap = np.mean(shap_values[:, feature_idx])

#                     direction = "↑ Higher" if mean_shap > 0 else "↓ Lower"

#                     # Categorize importance
#                     max_importance = max([f.get('importance', 0) for f in top_features]) if top_features else 1
#                     if max_importance > 0:
#                         relative_importance = importance / max_importance
#                         if relative_importance >= 0.66:
#                             magnitude = "High"
#                         elif relative_importance >= 0.33:
#                             magnitude = "Moderate"
#                         else:
#                             magnitude = "Low"

#                 except Exception as e:
#                     print(f"⚠️  Error determining direction for {feature}: {e}")

#             # Get clinical context
#             context = self._get_feature_context(feature)

#             summary_lines.append(
#                 f"{i}. {feature}: {direction} values ({magnitude} importance, SHAP={importance:.4f}). Clinical meaning: {context}")
#             # summary_lines.append(f"   Clinical meaning: {context}")
#             summary_lines.append("")  # Empty line for readability
            
#             print(f'feature_data: {feature_data}')
#             print(f'summary_lines: {summary_lines}')

#         return "\n".join(summary_lines)
    
    def generate_phenotype_interpretation(self, cluster_analysis: Dict, shap_values: np.ndarray = None, feature_indices: Dict = None) -> Dict:
        """
        Generate LLM-based interpretations for each phenotype with enhanced feature context using SHAP
        """
        print("Generating LLM-based phenotype interpretations with SHAP-based feature importance...")
        
        interpretations = {}
        
        for cluster_name, cluster_data in cluster_analysis.items():
            print(f"Interpreting {cluster_name}...")
            
            # Prepare physiological profile summary
            top_features = cluster_data['top_discriminative_features']
            n_patients = cluster_data['n_patients']
            percentage = cluster_data['percentage']
            
            # Get cluster-specific SHAP values and directions
            cluster_shap = cluster_data.get('shap_values', shap_values)
            # Get the pre-calculated feature directions for this cluster
            feature_directions = cluster_data.get('feature_directions', {}) # Fallback to empty dict if not present

            # Create enhanced feature summary with clinical context
            feature_summary = self._create_enhanced_feature_summary(top_features, cluster_shap, feature_indices, feature_directions)
            
#             prompt = f"""
#             Analyze this sepsis physiophenotype based on physiological monitoring data collected 5 minutes prior to sepsis-3 onset and interpret its physiological state based on sepsis pathophysiology:

#             CLUSTER DEMOGRAPHICS:
#             - Number of patients: {n_patients}
#             - Percentage of cohort: {percentage:.1f}%

#             TOP DISCRIMINATIVE PHYSIOLOGICAL FEATURES WITH CLINICAL CONTEXT:
#             {feature_summary}

#             PHYSIOLOGICAL SYSTEM CONTEXT:
#             - ECG features: Reflect cardiac electrical activity, systole phases, rhythm, QT prolongation, T-wave abnormalities and AF
#             - HRV features: Indicate autonomic nervous system function and stress response
#             - PPG features: Show peripheral circulation, perfusion and arterial stiffness
#             - Respiratory features: Indicate breathing patterns and respiratory mechanics
#             - qSOFA features: Clinical proxy for organ dysfunction risk in sepsis (SBP ≤ 100, RR ≥ 22, altered mentation)
            
#             SEPSIS PATHOPHYSIOLOGY CONSIDERATIONS:
#             - Autonomic dysfunction is common in sepsis (reduced HRV patterns and impaired baroreflex sensitivity)
#             - Cardiovascular compromise includes sepsis-induced cardiomyopathy and impaired contractility (ECG features like QT dynamics, QRS changes, arrhythmias)
#             - Hemodynamic instability & vasodilation come with reduced vascular tone and distributive shock (PPG AC amplitude, pulse arrival time, qSOFA_sbp, SBP)
#             - Respiratory dysfunction ranges from compensatory tachypnea to respiratory failure and ARDS (respiratory rate, qSOFA_rr)
#             - Vascular & microcirculatory collapse leads to impaired oxygen delivery and poor tissue perfusion (PPG waveform morphology)
#             - Systemic inflammation affects all physiological systems, contributing to multi-organ dysfunction (qSOFA)
#             - Compensated autonomic perturbation is an early or stable state of physiological stress, in which the body maintains vital signs within near-normal ranges (e.g., qSOFA = 0) despite underlying autonomic nervous system imbalance (e.g., tachycardia, altered HRV LF/HF ratio, mild tachypnea)

#             Based on these discriminative physiological patterns, please provide:

#             1. PHENOTYPE NAME: A clinically meaningful title (4-6 words) that captures the dominant pathophysiological pattern

#             2. PATHOPHYSIOLOGICAL INTERPRETATION: 
#                - What specific sepsis mechanisms does this physiological pattern suggest?
#                - Which organ systems appear most affected?
#                - What does the autonomic function pattern indicate?

#             3. CLINICAL CHARACTERISTICS:
#                - What would patients in this phenotype likely present with clinically?
#                - What hemodynamic profile would be expected?
#                - What respiratory patterns might be observed?

#             4. THERAPEUTIC IMPLICATIONS:
#                - What treatment considerations might be most relevant?
#                - Which monitoring priorities should be emphasized?
#                - What therapeutic targets are suggested by these patterns?

#             5. PROGNOSTIC INSIGHTS:
#                - What might this phenotype suggest about disease severity?
#                - What complications might be more likely?
#                - What recovery patterns might be expected?

#             Format your response with clear section headers and focus on the clinical significance of the physiological patterns.
#             """
            
#             prompt = f"""
#             Analyze this sepsis physiophenotype based on physiological monitoring data collected 5 minutes prior to sepsis-3 onset and interpret its physiological state based on sepsis pathophysiology:

#             CLUSTER DEMOGRAPHICS:
#             - Number of patients: {n_patients}
#             - Percentage of cohort: {percentage:.1f}%

#             TOP DISCRIMINATIVE PHYSIOLOGICAL FEATURES WITH CLINICAL CONTEXT:
#             {feature_summary}

#             PHYSIOLOGICAL SYSTEM CONTEXT:
#             - ECG features: Reflect cardiac electrical activity, systole phases, rhythm, QT prolongation, T-wave abnormalities and AF
#             - HRV features: Indicate autonomic nervous system function and stress response (e.g., ↓ Lower HRV metrics often indicate dysfunction/stress)
#             - PPG features: Show peripheral circulation, perfusion and arterial stiffness
#             - Respiratory features: Indicate breathing patterns and respiratory mechanics
#             - qSOFA features: Clinical proxy for organ dysfunction risk in sepsis (SBP ≤ 100, RR ≥ 22, altered mentation)

#             SEPSIS PATHOPHYSIOLOGY CONSIDERATIONS:
#             - Autonomic dysfunction is common in sepsis (e.g., ↓ Lower HRV patterns like SDNN, HF power, Complexity measures (S, ShanEn, CD), and DFA alpha indicate impaired autonomic regulation and stress).
#             - Cardiovascular compromise includes sepsis-induced cardiomyopathy and impaired contractility (ECG features like QT dynamics, QRS changes, arrhythmias).
#             - Hemodynamic instability & vasodilation come with reduced vascular tone and distributive shock (PPG AC amplitude, pulse arrival time, qSOFA_sbp, SBP).
#             - Respiratory dysfunction ranges from compensatory tachypnea to respiratory failure and ARDS (respiratory rate, qSOFA_rr).
#             - Vascular & microcirculatory collapse leads to impaired oxygen delivery and poor tissue perfusion (PPG waveform morphology).
#             - Systemic inflammation affects all physiological systems, contributing to multi-organ dysfunction (qSOFA).
#             - Compensated autonomic perturbation is an early or stable state of physiological stress, in which the body maintains vital signs within near-normal ranges (e.g., qSOFA = 0) despite underlying autonomic nervous system imbalance (e.g., tachycardia, altered HRV LF/HF ratio, mild tachypnea).

#             IMPORTANT INTERPRETATION GUIDELINES:
#             - Pay close attention to whether each feature value is "↑ Higher" or "↓ Lower" relative to the overall cohort.
#             - Use the specific "Clinical meaning" provided for each feature to understand the physiological implication of its change.
#             - For example, if HRV_HRV_SDNN is "↓ Lower values (High importance...)", interpret this as reduced overall HRV and autonomic balance, indicating dysfunction.
#             - Similarly, if PPG_A_sp_mean is "↑ Higher values", interpret this as higher average systolic pulse amplitude, which might indicate changes in peripheral perfusion or arterial tone depending on the context.

#             Based on these discriminative physiological patterns (directions and specific clinical meanings), please provide:

#             1. PHENOTYPE NAME: A clinically meaningful title (4-6 words) that captures the dominant pathophysiological pattern

#             2. PATHOPHYSIOLOGICAL INTERPRETATION: 
#                - What specific sepsis mechanisms does this pattern of HIGHER/LOWER features suggest? Focus on the specific metrics mentioned.
#                - Which organ systems appear most affected based on the specific features and their changes?
#                - What does the specific autonomic function pattern (based on detailed HRV metrics) indicate?

#             3. CLINICAL CHARACTERISTICS:
#                - What would patients in this phenotype likely present with clinically, based on the specific feature changes?
#                - What hemodynamic profile would be expected?
#                - What respiratory patterns might be observed?

#             4. THERAPEUTIC IMPLICATIONS:
#                - What treatment considerations might be most relevant, given the specific physiological imbalances?
#                - Which monitoring priorities should be emphasized based on the identified features?
#                - What therapeutic targets are suggested by these specific patterns?

#             5. PROGNOSTIC INSIGHTS:
#                - What might this specific phenotype suggest about disease severity?
#                - What complications might be more likely based on the physiological state described by the features?
#                - What recovery patterns might be expected?

#             Format your response with clear section headers and focus on the clinical significance of the specific physiological patterns and their directions.
#             """
        
       

            prompt = f"""
            Analyze this sepsis physiophenotype based on physiological monitoring data collected 5 minutes prior to sepsis-3 onset and interpret its physiological state based on sepsis pathophysiology:

            CLUSTER DEMOGRAPHICS:
            - Number of patients: {n_patients}
            - Percentage of cohort: {percentage:.1f}%

            TOP DISCRIMINATIVE PHYSIOLOGICAL FEATURES WITH CLINICAL CONTEXT:
            {feature_summary}

            PHYSIOLOGICAL SYSTEM CONTEXT:
            - ECG features: Reflect cardiac electrical activity, systole phases, rhythm, QT prolongation, T-wave abnormalities and AF
            - HRV features: Indicate autonomic nervous system function and stress response (e.g., ↓ Lower HRV metrics often indicate dysfunction/stress)
            - PPG features: Show peripheral circulation, perfusion and arterial stiffness
            - Respiratory features: Indicate breathing patterns and respiratory mechanics
            - qSOFA features: Clinical proxy for organ dysfunction risk in sepsis (SBP ≤ 100, RR ≥ 22, altered mentation)

            SEPSIS PATHOPHYSIOLOGY CONSIDERATIONS:
            - Autonomic dysfunction is common in sepsis (e.g., ↓ Lower HRV patterns like SDNN, HF power, Complexity measures (S, ShanEn, CD), and DFA alpha indicate impaired autonomic regulation and stress).
            - Cardiovascular compromise includes sepsis-induced cardiomyopathy and impaired contractility (ECG features like QT dynamics, QRS changes, arrhythmias).
            - Hemodynamic instability & vasodilation come with reduced vascular tone and distributive shock (PPG AC amplitude, pulse arrival time, qSOFA_sbp, SBP).
            - Respiratory dysfunction ranges from compensatory tachypnea to respiratory failure and ARDS (respiratory rate, qSOFA_rr).
            - Vascular & microcirculatory collapse leads to impaired oxygen delivery and poor tissue perfusion (PPG waveform morphology).
            - Systemic inflammation affects all physiological systems, contributing to multi-organ dysfunction (qSOFA).
            - Compensated autonomic perturbation is an early or stable state of physiological stress, in which the body maintains vital signs within near-normal ranges (e.g., qSOFA = 0) despite underlying autonomic nervous system imbalance (e.g., tachycardia, altered HRV LF/HF ratio, mild tachypnea).

            IMPORTANT INTERPRETATION GUIDELINES:
            - Pay close attention to whether each feature value is "↑ Higher" or "↓ Lower" relative to the overall cohort.
            - Use the specific "Clinical meaning" provided for each feature to understand the physiological implication of its change.
            - For example, if HRV_HRV_SDNN is "↓ Lower values (High importance...)", interpret this as reduced overall HRV and autonomic balance, indicating dysfunction.
            - Similarly, if PPG_A_sp_mean is "↑ Higher values", interpret this as higher average systolic pulse amplitude, which might indicate changes in peripheral perfusion or arterial tone depending on the context.

            Based on these discriminative physiological patterns (directions and specific clinical meanings), please provide:

            1. PHENOTYPE NAME: A clinically meaningful title (4-6 words) that captures the DISTINCTIVE pathophysiological pattern of THIS specific cluster. Focus on the specific combination of HIGH/LOW features mentioned in the 'TOP DISCRIMINATIVE PHYSIOLOGICAL FEATURES' section. For example, if the top features are predominantly ↓ Lower HRV Complexity (S, FuzzyEn) and ↑ Higher PPG Amplitude (A_AC), the name might be 'Reduced Complexity with Enhanced Pulse Strength'. If top features are ↑ Higher pNN50 (parasympathetic tone) and ↑ Higher HRV_ShanEn (complexity), it might be 'Enhanced Parasympathetic Complexity'. Avoid generic terms like "Dysfunction" or "Compromise" if the specific features suggest a more nuanced state.

            2. PATHOPHYSIOLOGICAL INTERPRETATION: 
               - What specific sepsis mechanisms does this pattern of HIGHER/LOWER features suggest? Focus on the specific metrics mentioned.
               - Which organ systems appear most affected based on the specific features and their changes?
               - What does the specific autonomic function pattern (based on detailed HRV metrics) indicate?

            3. CLINICAL CHARACTERISTICS:
               - What would patients in this phenotype likely present with clinically, based on the specific feature changes?
               - What hemodynamic profile would be expected?
               - What respiratory patterns might be observed?

            4. THERAPEUTIC IMPLICATIONS:
               - What treatment considerations might be most relevant, given the specific physiological imbalances?
               - Which monitoring priorities should be emphasized based on the identified features?
               - What therapeutic targets are suggested by these specific patterns?

            5. PROGNOSTIC INSIGHTS:
               - What might this specific phenotype suggest about disease severity?
               - What complications might be more likely based on the physiological state described by the features?
               - What recovery patterns might be expected?

            Format your response with clear section headers and focus on the clinical significance of the specific physiological patterns and their directions.
            """
            try:
                llm_response = self._call_llm(prompt)
                phenotype_name = self._extract_phenotype_name(llm_response)
                
                interpretations[cluster_name] = {
                    'phenotype_name': phenotype_name,
                    'full_interpretation': llm_response,
                    'cluster_stats': {
                        'n_patients': n_patients,
                        'percentage': percentage
                    },
                    'key_features': [f['feature'] for f in top_features[:5]],
                    'feature_contexts': [self._get_feature_context(f['feature']) for f in top_features[:5]]
                }
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error interpreting {cluster_name}: {e}")
                interpretations[cluster_name] = {
                    'phenotype_name': f'Phenotype {cluster_name.split("_")[1]}',
                    'full_interpretation': 'Interpretation unavailable due to API error',
                    'cluster_stats': {
                        'n_patients': n_patients,
                        'percentage': percentage
                    },
                    'key_features': [f['feature'] for f in top_features[:5]],
                    'feature_contexts': ['Context unavailable'] * 5
                }
        
        return interpretations
    
    def _extract_phenotype_name(self, llm_response: str) -> str:
        """Extract phenotype name from LLM response"""
        # Words that indicate the extraction likely grabbed the prompt structure
        prompt_keywords = {"name", "phenotype name", "phenotype", "title", "1.", "### 1. pheno", "### 1. name", "pheno", "### 1."}

        # Look for common patterns in LLM responses
        # Order is important: most specific and reliable patterns first
        patterns = [
            # Pattern 1: Look for quoted name immediately after "PHENOTYPE NAME:" (most reliable if quotes are used)
            # Handles: "PHENOTYPE NAME: "Name Here"" or "### 1. PHENOTYPE NAME: "Name Here""
            r"PHENOTYPE NAME\s*:\s*\"([^\"]+)\"",
            # Pattern 2: Look for quoted name after markdown header and PHENOTYPE NAME
            r"###\s*\d+\.\s*PHENOTYPE NAME\s*:\s*\"([^\"]+)\"", # e.g., "### 1. PHENOTYPE NAME: "Name""
            # Pattern 3: Look for any quoted name anywhere (fallback for quotes)
            r"\"([^\"]+)\"",
            # Pattern 4: Look for text after "PHENOTYPE NAME:" up to the end of the line, assuming it's the name.
            # This is less reliable but catches cases where the name isn't quoted.
            # It will capture everything after the colon until a newline.
            r"PHENOTYPE NAME\s*:\s*([^\n]+)",
            # Pattern 5: Look for text after markdown header "### 1." up to the end of the line, if it contains "PHENOTYPE NAME"
            # This captures the whole line if it starts with the header and contains the NAME, then we extract the part after ':'
            # This is a more generic catch for the first line if it's structured like the header.
            # We might need to post-process this if it captures the whole line.
            # r"###\s*1\.\s*([^\n]+)", # Keep this commented or use carefully, as it might catch other things.

        ]

        for pattern in patterns:
            matches = re.findall(pattern, llm_response, re.IGNORECASE)
            for match in matches:
                # match is the content captured by the parentheses in the regex
                name = match.strip()
                # If the pattern captured quotes, they should already be excluded by the group (?:...")

                # Check if the extracted name is likely a prompt keyword or header fragment
                # Increased character limit from 50 to 100, and word limit from 15 to 15 (or adjust as needed)
                if name.lower() not in prompt_keywords and 3 < len(name) < 100 and len(name.split()) <= 15:
                    print(f"Extracted name using pattern '{pattern}': '{name}'") # Debug print
                    return name
                else:
                    print(f"Skipped potential name (too generic/long/match): '{name}' from pattern '{pattern}'") # Debug print

        # Fallback: try to find a descriptive phrase not starting with prompt keywords
        # This is the same as before, but also applies the increased limits
        lines = llm_response.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            clean_line = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', line).strip() # Remove leading/trailing punctuation like ###
            # Remove potential header parts like "1. PHENOTYPE NAME:" or just "PHENOTYPE NAME:"
            clean_line = re.sub(r'^\d+\.\s*', '', clean_line) # Remove leading "1. ", "2. ", etc.
            clean_line = re.sub(r'^PHENOTYPE NAME\s*:\s*', '', clean_line, flags=re.IGNORECASE) # Remove leading "PHENOTYPE NAME:"
            clean_line = clean_line.strip() # Strip again after removing prefixes

            if (clean_line.lower() not in prompt_keywords and
                any(word in clean_line.lower() for word in ['pattern', 'type', 'profile', 'syndrome', 'dysfunction', 'instability', 'compromise']) and
                3 < len(clean_line) < 100 and len(clean_line.split()) <= 15): # Apply increased limits
                print(f"Extracted name using fallback line parsing: '{clean_line}'") # Debug print
                return clean_line

        # If all specific patterns fail, try the old fallback of capturing after "1. " again,
        # but be more careful to remove "PHENOTYPE NAME:" if it's captured.
        for line in lines[:10]:
            if line.strip().startswith("1."):
                 # Extract content after "1."
                 potential_name = re.sub(r'^\d+\.\s*', '', line).strip()
                 # Remove "PHENOTYPE NAME:" prefix if present
                 potential_name = re.sub(r'^PHENOTYPE NAME\s*:\s*', '', potential_name, flags=re.IGNORECASE).strip()
                 # Remove leading/trailing quotes again
                 potential_name = potential_name.strip('"\'')
                 if (potential_name.lower() not in prompt_keywords and
                     3 < len(potential_name) < 100 and len(potential_name.split()) <= 15): # Apply increased limits
                     print(f"Extracted name using final fallback (after '1.'): '{potential_name}'") # Debug print
                     return potential_name

        # If all else fails, return a default
        print("All extraction attempts failed, returning default name.") # Debug print
        return "Unnamed Phenotype"


    
  
    
#     def _extract_phenotype_name(self, llm_response: str) -> str:
#         """Extract phenotype name from LLM response"""
#         # Words that indicate the extraction likely grabbed the prompt structure
#         prompt_keywords = {"name", "phenotype name", "phenotype", "title", "1.", "### 1. pheno", "### 1. name", "pheno", "### 1."}

#         # Look for common patterns in LLM responses
#         # Order is important: more specific patterns first (e.g., quoted names, name after specific headers)
#         patterns = [
#             # Pattern 1: Look for PHENOTYPE NAME: followed by an optional quote, then capture the name (quoted or unquoted) until newline or ### section
#             # This handles cases like "### 1. PHENOTYPE NAME: "Name Here" ### 2. ..." or "### 1. PHENOTYPE NAME: Name Here ### 2. ..."
#             # It attempts to capture only the name part, potentially including quotes, but stops before the next section header.
#             r"PHENOTYPE NAME\s*:\s*\"?([^\n\"#]+(?:\"|(?=\s*###\s*\d\.)))", # Attempt to handle quotes and stop at next header
#             # Pattern 2: Look for NAME: followed by an optional quote, then capture the name until newline or ### section
#             r"NAME\s*:\s*\"?([^\n\"#]+(?:\"|(?=\s*###\s*\d\.)))", # Attempt to handle quotes and stop at next header
#             # Pattern 3: Look for quoted name after a section header line like "### 1. PHENOTYPE NAME:"
#             # This finds lines starting with ### 1. PHENOTYPE NAME: "Actual Name"
#             r"###\s*1\.\s*PHENOTYPE NAME\s*:\s*\"([^\"]+)\"", # Look specifically after 1. PHENOTYPE NAME for quoted text
#             # Pattern 4: Look for any quoted text (fallback for quotes)
#             r"\"([^\"]+)\"",
#             # Pattern 5: Look after the specific "### 1. PHENOTYPE NAME:" header, capturing the rest of the line, then stripping quotes
#             # This handles "### 1. PHENOTYPE NAME: "Name Here"" or "### 1. PHENOTYPE NAME: Name Here"
#             r"###\s*1\.\s*PHENOTYPE NAME\s*:\s*([^\n]+)",
#             # Pattern 6: Look after the general "PHENOTYPE NAME:" header, capturing the rest of the line, then stripping quotes
#             r"PHENOTYPE NAME\s*:\s*([^\n]+)",
#             # Pattern 7: Look for the content after "1. " (generic section 1)
#             r"1\.\s*([^\n]+)",
#         ]

#         for pattern in patterns:
#             matches = re.findall(pattern, llm_response, re.IGNORECASE)
#             for match in matches:
#                 # match could be the full string including quotes depending on the pattern's capture group
#                 name = match.strip()
#                 # If the pattern captured quotes as part of the group, remove them.
#                 # This is particularly relevant for patterns 1, 2, 5, 6 if they don't specifically exclude quotes in the group.
#                 name = name.strip('"\'') # Remove leading/trailing quotes

#                 # Check if the extracted name is likely a prompt keyword or header fragment
#                 if name.lower() not in prompt_keywords and 3 < len(name) < 50 and len(name.split()) <= 10:
#                     print(f"Extracted name using pattern '{pattern}': '{name}'") # Debug print
#                     return name
#                 else:
#                     print(f"Skipped potential name (too generic/long/match): '{name}' from pattern '{pattern}'") # Debug print

#         # Fallback: try to find a descriptive phrase not starting with prompt keywords
#         lines = llm_response.split('\n')
#         for line in lines[:10]:  # Check first 10 lines
#             clean_line = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', line).strip() # Remove leading/trailing punctuation like ###
#             # Remove potential header parts like "1. PHENOTYPE NAME:" or just "PHENOTYPE NAME:"
#             clean_line = re.sub(r'^\d+\.\s*', '', clean_line) # Remove leading "1. ", "2. ", etc.
#             clean_line = re.sub(r'^PHENOTYPE NAME\s*:\s*', '', clean_line, flags=re.IGNORECASE) # Remove leading "PHENOTYPE NAME:"
#             clean_line = clean_line.strip() # Strip again after removing prefixes

#             if (clean_line.lower() not in prompt_keywords and
#                 any(word in clean_line.lower() for word in ['pattern', 'type', 'profile', 'syndrome', 'dysfunction', 'instability', 'compromise']) and
#                 3 < len(clean_line) < 50):
#                 print(f"Extracted name using fallback line parsing: '{clean_line}'") # Debug print
#                 return clean_line

#         # If all else fails, return a default
#         print("All extraction attempts failed, returning default name.") # Debug print
#         return "Unnamed Phenotype"

    
#     def _extract_phenotype_name(self, llm_response: str) -> str:
#         """Extract phenotype name from LLM response"""
#         # Look for common patterns in LLM responses
#         patterns = [
#             r"NAME[:\s]*([^\n]+)",
#             r"PHENOTYPE NAME[:\s]*([^\n]+)",
#             r"1\.\s*([^\n]+)",
#             r"\"([^\"]+)\"",
#             r"'([^']+)'"
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, llm_response, re.IGNORECASE)
#             if match:
#                 name = match.group(1).strip()
#                 # Clean up the name
#                 name = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', name)  # Remove leading/trailing punctuation
#                 if len(name) > 3 and len(name.split()) <= 6:  # Reasonable length
#                     return name
        
#         # Fallback: try to find a descriptive phrase
#         lines = llm_response.split('\n')
#         for line in lines[:10]:  # Check first 10 lines
#             if any(word in line.lower() for word in ['phenotype', 'pattern', 'type', 'profile']):
#                 clean_line = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', line).strip()
#                 if 3 < len(clean_line) < 50:
#                     return clean_line
        
#         return "Unnamed Phenotype"
    
    def create_comprehensive_report(self, 
                                  data: pd.DataFrame,
                                  cluster_labels: np.ndarray,
                                  feature_columns: List[str],
                                  shap_values: np.ndarray = None,
                                  feature_indices: Dict = None,
                                  output_file: str = None) -> str:
        """
        Create a comprehensive interpretation report using SHAP-based feature importance
        
        Args:
            data: Patient data DataFrame
            cluster_labels: Cluster assignments
            feature_columns: List of feature names
            shap_values: SHAP values from clustering model
            feature_indices: Dictionary mapping feature names to indices in SHAP array
            output_file: Optional file path to save report
            
        Returns:
            Formatted report string
        """
        print("Creating comprehensive phenotype interpretation report using SHAP...")
        
        # Perform cluster analysis with SHAP
        cluster_analysis = self.analyze_cluster_characteristics(
            data, cluster_labels, feature_columns, shap_values, feature_indices)
        
        # Generate LLM interpretations
        interpretations = self.generate_phenotype_interpretation(
            cluster_analysis, shap_values, feature_indices)
        
        # Store interpretations
        self.phenotype_interpretations = interpretations
        
        # Create report
        report = self._format_comprehensive_report(cluster_analysis, interpretations, 
                                                 len(data), len(feature_columns))
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def _format_comprehensive_report(self, 
                                   cluster_analysis: Dict,
                                   interpretations: Dict,
                                   total_patients: int,
                                   total_features: int) -> str:
        """Format the comprehensive report"""
        
        report_sections = [
            "="*80,
            "SEPSIS PHYSIOPHENOTYPE INTERPRETATION REPORT",
            "="*80,
            "",
            f"Analysis Overview:",
            f"- Total patients: {total_patients:,}",
            f"- Total physiological features: {total_features}",
            f"- Number of identified phenotypes: {len(interpretations)}",
            f"- Analysis method: LLM-enhanced consensus clustering with SHAP",
            f"- Data window: 5 minutes prior to sepsis-3 onset",
            "",
            "="*80,
            "IDENTIFIED SEPSIS PHYSIOPHENOTYPES",
            "="*80,
            ""
        ]
        
        for cluster_name in sorted(interpretations.keys()):
            interp = interpretations[cluster_name]
            
            report_sections.extend([
                f"PHENOTYPE: {interp['phenotype_name']}",
                "-" * 60,
                f"Cluster ID: {cluster_name}",
                f"Patient Count: {interp['cluster_stats']['n_patients']} ({interp['cluster_stats']['percentage']:.1f}%)",
                f"Key Discriminative Features: {', '.join(interp['key_features'])}",
                "",
                "LLM CLINICAL INTERPRETATION:",
                interp['full_interpretation'],
                "",
                "="*60,
                ""
            ])
        
        # Add summary section
        phenotype_summary = []
        for cluster_name, interp in interpretations.items():
            phenotype_summary.append(
                f"• {interp['phenotype_name']}: {interp['cluster_stats']['n_patients']} patients "
                f"({interp['cluster_stats']['percentage']:.1f}%)"
            )
        
        report_sections.extend([
            "PHENOTYPE DISTRIBUTION SUMMARY:",
            "-" * 40,
            *phenotype_summary,
            "",
            "="*80,
            "ANALYSIS METADATA",
            "="*80,
            f"Generated using: {self.api_provider.upper()} {self.model_name}",
            f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Feature importance method: SHAP (SHapley Additive exPlanations)",
            "",
            "Note: This analysis uses AI-assisted interpretation of physiological patterns.",
            "Clinical validation and expert review are recommended before clinical application.",
            "="*80
        ])
        
        return "\n".join(report_sections)
    
    def visualize_phenotype_characteristics(self, 
                                          data: pd.DataFrame,
                                          cluster_labels: np.ndarray,
                                          feature_columns: List[str],
                                          shap_values: np.ndarray = None,
                                          feature_indices: Dict = None,
                                          save_path: str = None):
        """
        Create visualizations of phenotype characteristics using SHAP values
        
        Args:
            data: Patient data DataFrame
            cluster_labels: Cluster assignments
            feature_columns: List of feature names
            shap_values: SHAP values from clustering model
            feature_indices: Dictionary mapping feature names to indices in SHAP array
            save_path: Optional path to save figures
        """
        print("Creating phenotype characteristic visualizations using SHAP...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sepsis Physiophenotype Characteristics', fontsize=16, fontweight='bold')
        
        # 1. Phenotype distribution
        phenotype_counts = pd.Series(cluster_labels).value_counts().sort_index()
        phenotype_names0 = [self.phenotype_interpretations.get(f'Cluster_{i}', {}).get('phenotype_name', f'Phenotype {i}') 
                          for i in phenotype_counts.index]
        alphabets = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H']
        phenotype_names = [f'Phenotype {alphabets[i]}' for i in phenotype_counts.index]
        
        axes[0, 0].pie(phenotype_counts.values, labels=phenotype_names, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Phenotype Distribution')
        
        # 2. Feature importance heatmap for top features using SHAP
        cluster_analysis = self.analyze_cluster_characteristics(
            data, cluster_labels, feature_columns, shap_values, feature_indices)
        
        # Get top features across all clusters
        all_top_features = set()
        for cluster_data in cluster_analysis.values():
            top_features = [f['feature'] for f in cluster_data['top_discriminative_features'][:5]]
            all_top_features.update(top_features)
        
        # Create heatmap data using SHAP importance
        heatmap_data = []
        cluster_names = []
        
        for cluster_name, cluster_data in cluster_analysis.items():
            cluster_id = int(cluster_name.split('_')[1])
            phenotype_name = self.phenotype_interpretations.get(cluster_name, {}).get('phenotype_name', f'P{cluster_id}')
            cluster_names.append(phenotype_name)
            
            feature_values = []
            for feature in sorted(all_top_features):
                importance = next((f['importance'] for f in cluster_data['top_discriminative_features'] 
                                 if f['feature'] == feature), 0)
                feature_values.append(importance)
            heatmap_data.append(feature_values)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, 
                                    index=phenotype_names, #cluster_names,
                                    columns=sorted(all_top_features))
            
            sns.heatmap(heatmap_df, annot=True, cmap='YlOrRd', 
                       fmt='.3f', ax=axes[0, 1], cbar_kws={'label': 'SHAP Importance'})
            axes[0, 1].set_title('SHAP Feature Importance by Phenotype')
            axes[0, 1].tick_params(axis='x', rotation=90)
        
        # 3. Sample size comparison
        phenotype_sizes = [cluster_analysis[f'Cluster_{i}']['n_patients'] for i in phenotype_counts.index]
        bars = axes[1, 0].bar(range(len(phenotype_names)), phenotype_sizes, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(phenotype_names))))
        axes[1, 0].set_title('Sample Sizes by Phenotype')
        axes[1, 0].set_xlabel('Phenotype')
        axes[1, 0].set_ylabel('Number of Patients')
        axes[1, 0].set_xticks(range(len(phenotype_names)))
        axes[1, 0].set_xticklabels(phenotype_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, phenotype_sizes):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                           f'{value}', ha='center', va='bottom')
        
        # 4. Feature category distribution
        # feature_categories = {}
        # for feature in feature_columns:
        #     category = 'Other'
        #     for cat_name, keywords in self.feature_categories.items():
        #         if any(keyword.lower() in feature.lower() for keyword in keywords):
        #             category = cat_name
        #             break
        #     feature_categories[category] = feature_categories.get(category, 0) + 1
        
        # 4. Feature category distribution (using broad prefixes)
        feature_categories = {}
        for feature in feature_columns:
            # Determine broad category based on prefix
            if feature.startswith('ECG_'):
                category = 'ECG'
            elif feature.startswith('HRV_'):
                category = 'HRV'
            elif feature.startswith('PPG_'):
                category = 'PPG'
            elif feature.startswith('RESP_'):
                category = 'RESP'
            elif feature.startswith('ABP_'):
                category = 'ABP'
            elif feature.startswith('qSOFA_'):
                category = 'qSOFA'
            else:
                category = 'Other' # For features not starting with the specified prefixes
            
            feature_categories[category] = feature_categories.get(category, 0) + 1
        
        axes[1, 1].pie(feature_categories.values(), labels=feature_categories.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Feature Category Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def get_phenotype_names(self) -> Dict[int, str]:
        """
        Get mapping of cluster IDs to phenotype names
        
        Returns:
            Dictionary mapping cluster ID to phenotype name
        """
        if not self.phenotype_interpretations:
            print("No interpretations available. Run create_comprehensive_report first.")
            return {}
        
        phenotype_names = {}
        for cluster_name, interp in self.phenotype_interpretations.items():
            cluster_id = int(cluster_name.split('_')[1])
            phenotype_names[cluster_id] = interp['phenotype_name']
        
        return phenotype_names
    
    def export_results(self, output_dir: str = "./sepsis_phenotype_analysis/"):
        """
        Export all results including interpretations, visualizations, and data
        
        Args:
            output_dir: Directory to save all outputs
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export interpretations as JSON
        json_path = os.path.join(output_dir, "phenotype_interpretations.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.phenotype_interpretations, f, indent=2, ensure_ascii=False)
        
        # Export phenotype name mapping
        names_path = os.path.join(output_dir, "phenotype_names.json")
        with open(names_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_phenotype_names(), f, indent=2)
        
        print(f"Results exported to: {output_dir}")


# Example usage function
def run_sepsis_phenotype_interpretation(data: pd.DataFrame,
                                      cluster_labels: np.ndarray,
                                      feature_columns: List[str],
                                      top_5_features_per_class: Dict,
                                      shap_values: np.ndarray = None,
                                      feature_indices: Dict = None,
                                      api_provider: str = 'openai',
                                      api_key: str = None,
                                      model_name: str = None,
                                      output_dir: str = "./sepsis_analysis_results/"):
    """
    Complete pipeline for sepsis phenotype interpretation using SHAP
    
    Args:
        data: Patient data DataFrame (2074 patients x 400+ features)
        cluster_labels: Cluster assignments from consensus clustering
        feature_columns: List of physiological feature names
        shap_values: SHAP values from clustering model
        feature_indices: Dictionary mapping feature names to indices in SHAP array
        api_provider: 'openai' or 'anthropic'
        api_key: API key for chosen provider
        model_name: Specific model name
        output_dir: Output directory for results
    """
    
    # Initialize interpreter
    interpreter = SepsisPhysiophenotypeInterpreter(
        api_provider=api_provider,
        api_key=api_key,
        model_name=model_name
    )
    
    # Create comprehensive report using SHAP
    report = interpreter.create_comprehensive_report(
        data=data,
        cluster_labels=cluster_labels,
        feature_columns=feature_columns,
        shap_values=shap_values,
        feature_indices=feature_indices,
        output_file=f"{output_dir}/sepsis_phenotype_report.txt"
    )
    
    # Create visualizations using SHAP
    interpreter.visualize_phenotype_characteristics(
        data=data,
        cluster_labels=cluster_labels,
        feature_columns=feature_columns,
        shap_values=shap_values,
        feature_indices=feature_indices,
        save_path=f"{output_dir}/phenotype_characteristics.png"
    )
    
    # Export all results
    interpreter.export_results(output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SEPSIS PHENOTYPE ANALYSIS COMPLETE")
    print("="*60)
    
    phenotype_names = interpreter.get_phenotype_names()
    print("\nIdentified Sepsis Physiophenotypes:")
    for cluster_id, name in phenotype_names.items():
        n_patients = np.sum(cluster_labels == cluster_id)
        percentage = n_patients / len(cluster_labels) * 100
        print(f"• Cluster {cluster_id}: {name} ({n_patients} patients, {percentage:.1f}%)")
    
    print(f"\nAll results saved to: {output_dir}")
    
    return interpreter, report


# Example usage with your data:
"""
# Assuming you have:
# - df_data: DataFrame with patient data
# - cluster_labels: Array with cluster assignments from your consensus clustering
# - feature_columns: List of your physiological features
# - shap_values: SHAP values from the clustering pipeline
# - feature_indices: Dictionary mapping feature names to SHAP indices

# Set up your API credentials
API_KEY = "your-api-key-here"

# Run the complete analysis with SHAP
interpreter, report = run_sepsis_phenotype_interpretation(
    data=df_data,
    cluster_labels=cluster_labels,
    feature_columns=feature_columns,
    shap_values=shap_values,
    feature_indices=feature_indices,
    api_provider='openai',  # or 'anthropic'
    api_key=API_KEY,
    model_name='gpt-4o-mini',  # or 'claude-3-sonnet-20240229'
    output_dir="./sepsis_phenotype_results/"
)

# Access phenotype names
phenotype_names = interpreter.get_phenotype_names()
print("Phenotype Names:", phenotype_names)
"""

"""
Modified on Sat May 4 22:51:46 2024

@author: Tilendra Choudhary
"""

import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
import MPT as mpt
import wfdb
import pandas as pd
import argparse
import numpy as np

output_dir = './Output_Data/'
# Create the output directory if it doesn't exist
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def load_signal(path, name):
    if path is None:
        print(f"{name} signal not provided.")
        return []
    elif not os.path.exists(path):
        raise FileNotFoundError(f"{name} file '{path}' not found.")
    return np.load(path)

def main(ecg, ppg, resp, abp, fs, stat_method, Show_plotly, Show_matplotlib):
    mpt_process = mpt.physiomarkers_from_biosignals(ecg, ppg, resp, abp, fs, Show_plotly, Show_matplotlib)
    mpt_process.FiducialpointDetect_with_SQA(F_s=1000) #F_s: Resampling frequency
    #print(mpt_process.fiducials)
    features_ts = mpt_process.feature_extraction()
    feature_dict = mpt_process.summary_statistics(stat_method=stat_method) #stat_method: 'all', 'mean', 'median', 'min', 'max'
    return feature_dict, features_ts, mpt_process.fiducials, mpt_process.signals_filt


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Physiomarkers extraction from biosignals')
    parser.add_argument('--sample', action='store_true', help='Use sample BIDMC data (default: False)')
    parser.add_argument('--sig_duration_min', type=int, default=5, help='Sampling frequency of input signals')
    parser.add_argument('--show_plotly', action='store_true', help='Show Plotly plots')
    parser.add_argument('--show_matplotlib', action='store_true', help='Show matplotlib plots')
    parser.add_argument('--stat_method', type=str, default='median', help="Statistics method for feature extraction: 'all', 'mean', 'median', 'min', 'max' (default: Median)")
    parser.add_argument('--ecg', type=str, help='Path to ECG .npy file')
    parser.add_argument('--ppg', type=str, help='Path to PPG .npy file')
    parser.add_argument('--resp', type=str, help='Path to Resp .npy file')
    parser.add_argument('--abp', type=str, help='Path to ABP .npy file')
    parser.add_argument('--fs', type=int, default=None, help='Sampling frequency')
    args = parser.parse_args()

    if args.sample:
        FilePath = './Sample_Data/bidmc09m'
        record = wfdb.rdrecord(FilePath)
        signals = record.p_signal
        signal_names = record.sig_name
        print("Available signals:", signal_names)
    
        def extract_signal(name):
            try:
                idx = signal_names.index(name)
                print(f'{name.strip()} is present!')
                return signals[:, idx]
            except ValueError:
                print(f'{name.strip()} is not found!')
                return []
        # Accessing the signals: ECG, PPG, RESP, ABP
        ecg = extract_signal('II,')
        ppg = extract_signal('PLETH,')
        resp = extract_signal('RESP,')
        abp = extract_signal('ABP,')
        fs = record.fs

    else:
        ecg = load_signal(args.ecg, "ECG")
        ppg = load_signal(args.ppg, "PPG")
        resp = load_signal(args.resp, "RESP")
        abp = load_signal(args.abp, "ABP")
        fs = args.fs

    sig_len = args.sig_duration_min * 60 * fs
    ecg_s = ecg[:sig_len] if len(ecg) else []
    ppg_s = ppg[:sig_len] if len(ppg) else []
    resp_s = resp[:sig_len] if len(resp) else []
    abp_s = abp[:sig_len] if len(abp) else []

    if args.show_matplotlib:
        fig, axs = plt.subplots(4, 1, figsize=(21, 7))
        axs[0].plot(ecg_s); axs[0].set_ylabel('ECG'); axs[0].set_title('Raw waveforms')
        axs[1].plot(ppg_s); axs[1].set_ylabel('PPG')
        axs[2].plot(resp_s); axs[2].set_ylabel('Resp.')
        axs[3].plot(abp_s); axs[3].set_ylabel('ABP'); axs[3].set_xlabel('Sample number')
        for ax in axs: ax.autoscale(enable=True, axis='x', tight=True)
        plt.show()
    
    t1 = time.time()
    feature_dict, physiomarkers_ts, fiducial_timeSeries, filtered_signals = main(ecg_s, ppg_s, resp_s, abp_s, fs,
                                                                    stat_method=args.stat_method,
                                                                    Show_plotly=args.show_plotly,
                                                                    Show_matplotlib=args.show_matplotlib)
    t2 = time.time()
    print(f'Execution time: {t2 - t1:.2f} seconds')


    # Convert to DataFrame
    df_physiomarkers = pd.DataFrame([feature_dict])
    df_physiomarkers.to_csv(output_dir+'physiomarkers.csv', index=False)
    print(f'Physiomarkers saved to {output_dir}physiomarkers.csv')




    """
    =======================================================
    Example usage of the script:
    =======================================================
    python MPT_physiomarkers_wrapper.py --sample --show_matplotlib
    python MPT_physiomarkers_wrapper.py --sample --show_plotly
    python MPT_physiomarkers_wrapper.py --sample --show_plotly --show_matplotlib
    python MPT_physiomarkers_wrapper.py --sample --sig_duration_min 1
    python MPT_physiomarkers_wrapper.py --sample --sig_duration_min 2 --show_matplotlib
    python MPT_physiomarkers_wrapper.py --sample --sig_duration_min 5 --stat_method mean 
    python MPT_physiomarkers_wrapper.py --sample --sig_duration_min 5 --stat_method all 
    """
   
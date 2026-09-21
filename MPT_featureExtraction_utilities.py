import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt 
import neurokit2 as nk
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import iqr, skew, kurtosis, entropy as scipy_entropy



##FUNCTION: Noise Filtering
def butter_highpass(self, cutoff, fs, order=2):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a

def butter_lowpass(self, cutoff, fs, order=2):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a

def final_filter(self, data, fs, cutoff_high, cutoff_low, order=2):  ### filtfilt: forword and backword filtering to perform zero-phase filtering 
    b, a = self.butter_highpass(cutoff_high, fs, order=order)
    x = filtfilt(b, a, data)
    d, c = self.butter_lowpass(cutoff_low, fs, order = order)
    y = filtfilt(d, c, x)
    return y



##FUNCTIONS: Time-Frequency Decomposition (mostly useful for RESP preprocessing)
def signal_decomposition_based_preprocessing(self, signal, sampling_rate, show=False): 
    # Decompose signal using Empirical Mode Decomposition (EMD)
    components = nk.signal_decompose(signal, method='emd')
    # Recompose merging correlated components
    recomposed = nk.signal_recompose(components, threshold=0.99) 
    if show:
        nk.signal_plot(components)  # Visualize components
        nk.signal_plot(recomposed)  # Visualize components
    center_frequencies = []
    for i in range(recomposed.shape[0]):
        center_frequency = self.signal_center_frequency(recomposed[i], sampling_rate, show)
        center_frequencies.append(center_frequency)
    return recomposed, center_frequencies
        
    
    
    
def signal_center_frequency(self, signal, sampling_rate, show=False):
    ## Center frequency (CF), also known as center of gravity
    # welch frequency * power spectral density / summation of power spectral density
    welch = nk.signal_psd(signal, method="welch", sampling_rate=sampling_rate, min_frequency=0, max_frequency=sampling_rate/2, show=show)
    f = welch['Frequency']
    psd = welch['Power']
    numerator = np.sum(f * psd)
    denominator = np.sum(psd)
    center_frequency = numerator / denominator if denominator != 0 else 0
    return center_frequency



"""

Fiducial point functions

"""

##FUNCTION: Sanitization of detected fiducial points by retaining 99.7% data
def point_outlier_rejection(self, signal, point_locs, th=3):
    point_locs = np.unique(point_locs)
    point_series = signal[point_locs]
    point_array = np.array(point_series)
    point_indx1 = np.where(point_array<(np.mean(point_array)+th*np.std(point_array)))[0]
    point_indx2 = np.where(point_array>(np.mean(point_array)-th*np.std(point_array)))[0]
    point_indx = np.intersect1d(point_indx1, point_indx2) #common indexes
    point_locs_final = point_locs[point_indx]
    return pd.Series(point_locs_final)
    



##FUNCTION: Peak correction for peaks (peaktype=1) and troughs (peaktype=0)
def Peak_correction(self, peaks, signal, fs, peaktype=1,  window=150e-3):  #window=100e-3
    ptrue = np.zeros(len(peaks)) #[]
    k = 0
    for i in peaks:
        range1 = signal[i-int(np.floor(fs*window)):i+int(np.floor(fs*window))]
        if peaktype==1: mx = np.argmax(range1)
        else: mx = np.argmin(range1)
        ptrue[k] = i-int(np.floor(fs*window))+mx
        k = k+1
    pk_true = np.unique(ptrue)
    pk_true = list(pk_true.astype(int))
    return pk_true



##FUNCTION: PPG foot or Onset Detection
def PPG_foot(self, PPGpeaks, signal):
    foot = np.zeros(len(PPGpeaks))
    k = 0
    for i in range(len(PPGpeaks)-1):
        range1 = np.arange(PPGpeaks[i], PPGpeaks[i+1]+1)
        cycle  =  signal[range1]
        mn = np.argmin(cycle)
        foot[k] = PPGpeaks[i]+mn
        k = k+1
    foot = np.delete(foot, np.where(foot == 0))
    foot = list(foot.astype(int))
    return foot

##FUNCTION: Flatline boundary detection for raw signals
def tc_flatline_boundry_detect(self, signal, n):
    ### if samples are repeating for more than 'n' times consequtively (say 1 s for n)
    from itertools import groupby
    signal = pd.Series(signal)
    diff_signal = np.array(signal.diff()==0, dtype=int)
    ix = 0
    ixs_start = []
    ixs_end = []
    for k,g in groupby(diff_signal):
        leng = len(list(g))
        if k and leng >= n:
            ixs_start.append(ix-1)
            ixs_end.append(ix+leng+1)
        ix += leng
    ix_start_stop = list(zip(ixs_start, ixs_end))
    flatline_loc = np.zeros(len(signal))
    for i in range(len(ix_start_stop)):
        flatline_loc[ix_start_stop[i][0]:ix_start_stop[i][1]]=1
    return ix_start_stop, flatline_loc


##FUNCTION: Signal outlier correction
def signal_oulier_correction(self, signal, th=6):
    signal_samples1 = np.where(signal>(np.mean(signal)+th*np.std(signal)))
    signal_samples2 = np.where(signal<(np.mean(signal)-th*np.std(signal)))
    signal[signal_samples1] = np.median(signal)
    signal[signal_samples2] = np.median(signal)
    return signal



##FUNCTION: Fiducial point detection for PPG, VPG and APG signals
def ppg_fiducials_detection(self, ppg_filtered_signal, F_s, locs_peaks, locs_onsets):
    
    PPG_peak_amps = ppg_filtered_signal[locs_peaks]
    PPG_onset_amps= ppg_filtered_signal[locs_onsets]

    #Calculate first and second derivatives of the PPG signal
    vpg_sig = np.gradient(ppg_filtered_signal) / (1/F_s) 
    apg_sig = np.gradient(vpg_sig) / (1/F_s) 
    
    
    vpg_sig = self.signal_oulier_correction(vpg_sig, th=5)
    apg_sig = self.signal_oulier_correction(apg_sig, th=5)

    try:
        vpg_fiducials = vpg_delineate(vpg_sig, F_s)
        apg_fiducials = apg_delineate(apg_sig, vpg_sig, vpg_fiducials, F_s)
        ppg_fiducials = ppg_delineate(ppg_filtered_signal, vpg_sig, vpg_fiducials, 
                                      apg_sig, apg_fiducials, locs_peaks, locs_onsets, F_s)
    except TypeError: 
        vpg_fiducials = {'w_waves':[],'y_waves':[], 'z_waves':[]}
        apg_fiducials = {'a_waves':[], 'b_waves':[], 'c_waves':[], 'd_waves':[], 'e_waves':[]}
        ppg_fiducials = {'O_waves':[], 'S_waves':[], 'N_waves':[], 'D_waves':[]}
    except ValueError: 
        vpg_fiducials = {'w_waves':[],'y_waves':[], 'z_waves':[]}
        apg_fiducials = {'a_waves':[], 'b_waves':[], 'c_waves':[], 'd_waves':[], 'e_waves':[]}
        ppg_fiducials = {'O_waves':[], 'S_waves':[], 'N_waves':[], 'D_waves':[]}
    except IndexError:
        vpg_fiducials = {'w_waves':[],'y_waves':[], 'z_waves':[]}
        apg_fiducials = {'a_waves':[], 'b_waves':[], 'c_waves':[], 'd_waves':[], 'e_waves':[]}
        ppg_fiducials = {'O_waves':[], 'S_waves':[], 'N_waves':[], 'D_waves':[]}


    return locs_peaks, PPG_peak_amps, locs_onsets, PPG_onset_amps, vpg_sig, apg_sig, vpg_fiducials, apg_fiducials, ppg_fiducials




"""

Feature Extraction functions
Reference for PPG features: https://doi.org/10.1152/ajpheart.00392.2021

"""

'''UTILITIES'''

#### ------ PPG Fiducial Points ---------

def extrema_search_win_indices(w_len, sig_len):
    """Fiducials can be extracted with a extrema search window of w_len samples."""
    ind = []
    for i in range(0, sig_len - w_len + 1, w_len - 2):
        ind_ = np.arange(i, i + w_len, 1)
        ind.append(ind_)

    return ind


def identify_slope_reversals(
    signal,
    slope_direction,
    traversal_direction,
    reversal_criterion,
    search_ranges = None,
    segment_starts = None,
    segment_ends = None,
):
    """It looks for a slope reversal point in a provided signal."""

    if search_ranges is None:
        # Create search ranges using segment starts and ends
        search_ranges = [np.arange(start, end) for start, end in zip(segment_starts, segment_ends)]

    reversal_locations = []
    for range_idx in range(len(search_ranges)):
        index_range = search_ranges[range_idx]
        signal_segment = signal[index_range]

        reversal_indices = detect_slope_reversals(
            signal_segment,
            slope_type=slope_direction,
            scan_direction=traversal_direction,
            selection_rule=reversal_criterion,
        )
        if np.size(reversal_indices) != 0:
            actual_indices = index_range[reversal_indices]
            reversal_locations = np.append(reversal_locations, actual_indices).astype(int)

    return reversal_locations

def detect_slope_reversals(
    signal,
    slope_type = "both",
    scan_direction = "left_to_right",
    selection_rule = "all"
):
    """
    Detects slope reversal points in a 1D signal array according to specified slope type,
    scan direction, and selection rule.
    """
    # Compute the slope (first derivative) of the signal
    slope_values = np.diff(signal)

    # Determine slope reversal points based on slope type and scan direction
    if slope_type == "positive":
        if scan_direction == "left_to_right":
            reversal_indices = np.where((slope_values[:-1] < 0) & (slope_values[1:] >= 0))[0] + 1
        elif scan_direction == "right_to_left":
            reversal_indices = np.where((slope_values[:-1] >= 0) & (slope_values[1:] < 0))[0] + 1
        else:
            raise ValueError("Invalid scan direction provided.")

    elif slope_type == "negative":
        if scan_direction == "left_to_right":
            reversal_indices = np.where((slope_values[:-1] > 0) & (slope_values[1:] <= 0))[0] + 1
        elif scan_direction == "right_to_left":
            reversal_indices = np.where((slope_values[:-1] <= 0) & (slope_values[1:] > 0))[0] + 1
        else:
            raise ValueError("Invalid scan direction provided.")

    elif slope_type == "both":
        if scan_direction == "left_to_right":
            reversal_indices = np.where(np.sign(slope_values[:-1]) != np.sign(slope_values[1:]))[0] + 1
        elif scan_direction == "right_to_left":
            reversed_slopes = slope_values[::-1]
            reversal_indices = np.where(np.sign(reversed_slopes[:-1]) != np.sign(reversed_slopes[1:]))[0][::-1] + 1
        else:
            raise ValueError("Invalid scan direction provided.")

    else:
        raise ValueError("Invalid slope type provided.")

    # Apply selection rule to filter detected reversals
    if selection_rule == "all":
        return reversal_indices
    elif selection_rule == "first":
        return reversal_indices[0] if len(reversal_indices) > 0 else []
    elif selection_rule == "max":
        return reversal_indices[np.argmax(signal[reversal_indices])] if len(reversal_indices) > 0 else []
    elif selection_rule == "min":
        return reversal_indices[np.argmin(signal[reversal_indices])] if len(reversal_indices) > 0 else []
    else:
        raise ValueError("Invalid selection rule provided.")



def zero_crossings_detect(
    signal,
    crossing_type,
    scan_direction,
    selection_mode,
    segment_starts = None,
    segment_ends = None,
    scan_ranges = None,
):
    """
    Searches for zero crossing points in the given signal according to specified parameters.
    """
    if scan_ranges is None:
        # Generate scan index ranges using segment start and end positions
        scan_ranges = [np.arange(start, end) for start, end in zip(segment_starts, segment_ends)]

    crossings = []

    for segment_idx in range(len(segment_starts)):
        current_range = scan_ranges[segment_idx]
        crossings_found = []

        if scan_direction == "left_to_right":
            for i in range(len(current_range) - 1):
                idx = current_range[i]
                next_idx = current_range[i + 1]

                if crossing_type == "positive":
                    if (signal[idx] < 0) and (signal[next_idx] >= 0):
                        crossings_found.append([idx, signal[idx]])
                        if selection_mode == "first":
                            break

                elif crossing_type == "negative":
                    if (signal[idx] > 0) and (signal[next_idx] <= 0):
                        crossings_found.append([idx, signal[idx]])
                        if selection_mode == "first":
                            break

                elif crossing_type == "both":
                    if (signal[idx] == 0) and (signal[next_idx] != 0):
                        crossings_found.append([idx, signal[idx]])
                        if selection_mode == "first":
                            break

                else:
                    raise ValueError("Invalid crossing type specified.")

        elif scan_direction == "right_to_left":
            for i in reversed(range(len(current_range) - 1)):
                idx = current_range[i]
                next_idx = current_range[i + 1]

                if crossing_type == "positive":
                    if (signal[idx] < 0) and (signal[next_idx] >= 0):
                        crossings_found.append([idx, signal[idx]])
                        if selection_mode == "first":
                            break

                elif crossing_type == "negative":
                    if (signal[idx] > 0) and (signal[next_idx] <= 0):
                        crossings_found.append([idx, signal[idx]])
                        if selection_mode == "first":
                            break

                elif crossing_type == "both":
                    if (signal[idx] == 0) and (signal[next_idx] != 0):
                        crossings_found.append([idx, signal[idx]])
                        if selection_mode == "first":
                            break

                else:
                    raise ValueError("Invalid crossing type specified.")

        else:
            raise ValueError("Invalid scan direction specified.")

        if selection_mode == "last":
            if len(crossings_found) > 0:
                crossings.append(crossings_found[-1])
            else:
                crossings.append([len(signal), signal[-1]])
        else:
            crossings.extend(crossings_found)
    return crossings





''' 
The following functions, vpg_delineate(), apg_delineate(), ppg_delineate(), were implemented by following the methods presented in:
- Abhishek Chakraborty, Deboleena Sadhukhan & Madhuchhanda Mitra (2019): 
"An Automated Algorithm to Extract Time Plane Features From the PPG Signal and its Derivatives for Personal Health Monitoring Application," 
IETE Journal of Research, DOI: 10.1080/03772063.2019.1604178
'''


def vpg_delineate(signal, fs, threshold_w_peak=0.5, threshold_y_trough=0.45):
    """
    Delineates the VPG signal to extract fiducial points.

    Reference:
    Abhishek Chakraborty, Deboleena Sadhukhan & Madhuchhanda Mitra (2019): 
    An Automated Algorithm to Extract Time Plane Features From the PPG Signal and
    its Derivatives for Personal Health Monitoring Applications, IETE Journal of Research, 
    DOI: 10.1080/03772063.2019.1604178
    
    Parameters:
    - signal (ArrayLike): The VPG signal to be delineated.
    - fs (float): The sampling rate of the VPG signal.
    - threshold_w_peak (float): Threshold for detecting w waves. Defaults to 0.5.
    - threshold_y_trough (float): Threshold for detecting y waves. Defaults to 0.45.
    
    Returns:
    - wave_locations: A dictionary containing the delineated fiducial points.
    """

    
    
    wave_locations = {}

    #----------------
    # Extraction of w-waves (largest peak of VPG)
    #----------------

    peak_amplitude = np.max(signal)
    threshold_w = threshold_w_peak * peak_amplitude

    processed_signal = signal.copy()
    processed_signal[signal < threshold_w] = 0

    window_length = round(0.25 * fs)

    # Generate the search indices
    search_indices = extrema_search_win_indices(window_length, sig_len=len(processed_signal))

    # Search for a slope reversal point in each window (window_length)
    wave_peak_locations = identify_slope_reversals(
        processed_signal, slope_direction="negative", traversal_direction="left_to_right", reversal_criterion="all", search_ranges=search_indices
    )

    #----------------
    # Extraction of y-waves (lowest trough of VPG)
    #----------------

    trough_amplitude = min(signal)
    threshold_y = threshold_y_trough * trough_amplitude

    # Define a search interval
    search_start_points = wave_peak_locations
    locations = np.append(wave_peak_locations, len(signal))
    search_intervals = np.round(np.diff(locations) / 2).astype(int)
    search_end_points = search_start_points + search_intervals
    processed_signal = np.zeros(len(signal))

    for k in range(len(search_intervals)):
        processed_signal[search_start_points[k] : search_end_points[k]] = signal[search_start_points[k] : search_end_points[k]]

    processed_signal[signal > threshold_y] = 0

    # Generate the search indices
    search_indices = extrema_search_win_indices(window_length, sig_len=len(processed_signal))

    # Search for a slope reversal in each window (window_length)
    search_end_adjusted = np.append(wave_peak_locations[1:], len(signal) - 1)
    y_wave_locations = identify_slope_reversals(
        processed_signal,
        slope_direction="positive",
        traversal_direction="left_to_right",
        reversal_criterion="min",
        segment_starts=search_start_points,
        segment_ends=search_end_adjusted,
    )
    y_wave_peaks = signal[y_wave_locations]

    
    #----------------
    # Extraction of z-waves (local extreme)
    #----------------

    # search for the next zero crossing point
    search_start_points = y_wave_locations + 1
    search_end_points = np.append(search_start_points[1:], len(signal))
    # Detect zero crossings
    zero_crossing_points = zero_crossings_detect(
        signal,
        segment_starts=search_start_points,
        segment_ends=search_end_points,
        crossing_type="positive",
        scan_direction="left_to_right",
        selection_mode="first",
    )


    zero_cross_indices = np.array([x[0] for x in zero_crossing_points])

    # if number of y peaks not is matching with number of zero crossing points
    if len(zero_cross_indices) != len(y_wave_locations):
        if y_wave_locations[0] > zero_cross_indices[0]:
            zero_cross_indices = np.insert(zero_cross_indices, 0, np.mean(np.diff(zero_cross_indices)))

        if y_wave_locations[-1] > zero_cross_indices[-1]:
            zero_cross_indices = np.append(zero_cross_indices, len(signal) - 1)

    # modify the signal (first 70% percent of the samples)
    samples_to_modify = np.round(0.7 * (zero_cross_indices - y_wave_locations)).astype(int)

    start_indices = y_wave_locations
    y_wave_peaks = y_wave_peaks
    end_indices = y_wave_locations + samples_to_modify
    modified_y_wave_peaks = signal[end_indices]

    slope = (modified_y_wave_peaks - y_wave_peaks) / (end_indices - start_indices)

    modified_signal = signal.copy()

    max_value_indices = []
    for j in range(len(slope)):
        i = 0
        for k in range(start_indices[j], end_indices[j] + 1):
            modified_signal[k] = signal[k] - slope[j] * i - y_wave_peaks[j]
            i += 1

        # mod_max = np.max(modified_signal[start_indices[j]:end_indices[j]+1])  # Find the maximum
        max_index = np.argmax(modified_signal[start_indices[j] : end_indices[j] + 1])  # Find the index of maximum
        max_index += start_indices[j]
        max_value_indices.append(max_index)

    z_wave_locations = np.array(max_value_indices)

    wave_locations["w_waves"] = wave_peak_locations
    wave_locations["y_waves"] = y_wave_locations
    wave_locations["z_waves"] = z_wave_locations

    return wave_locations





def apg_delineate(
    apg_signal,
    vpg_signal,
    vpg_fiducials,
    sampling_rate,
    threshold_a = 0.45,
    threshold_w = 0.5,
):
    """Detects fiducials (key points) in the APG signal based on VPG signal fiducials.

    Reference:
    Abhishek Chakraborty, Deboleena Sadhukhan & Madhuchhanda Mitra (2019): 
    An Automated Algorithm to Extract Time Plane Features From the PPG Signal and
    its Derivatives for Personal Health Monitoring Applications, IETE Journal of Research, 
    DOI: 10.1080/03772063.2019.1604178

    Args:
        apg_signal (ArrayLike): The APG signal.
        vpg_signal (ArrayLike): The VPG signal.
        vpg_fiducials (dict): Dictionary containing VPG fiducials, like locations of y, z waves.
        sampling_rate (float): The sampling rate of the APG signal in Hz.
        threshold_a (float, optional): The threshold for detecting a-waves. Default is 0.45.
        threshold_w (float, optional): The threshold for detecting w-waves. Default is 0.5.

    Returns:
        dict: A dictionary containing the locations of key fiducials (a, b, c, d, e) in the APG signal.
    """
    
    # Get fiducials from the VPG signal
    vpg_y_waves = vpg_fiducials["y_waves"]
    vpg_z_waves = vpg_fiducials["z_waves"]

    fiducials = {}

    ###########################################
    # Detect a-waves (peak of the APG signal)
    ###########################################

    max_amplitude = np.max(apg_signal)
    threshold_a_value = threshold_a * max_amplitude

    processed_apg_signal = apg_signal.copy()
    processed_apg_signal[apg_signal < threshold_a_value] = 0

    window_length = round(0.25 * sampling_rate)

    threshold_w_value = threshold_w * np.max(vpg_signal)
    processed_vpg_signal = vpg_signal.copy()
    processed_vpg_signal[vpg_signal < threshold_w_value] = 0

    # Generate search indices for slope reversals in the VPG signal
    search_indices = extrema_search_win_indices(window_length, sig_len=len(processed_vpg_signal))

    # Detect slope reversal points in the APG signal for a-waves
    a_wave_locations = identify_slope_reversals(
        processed_apg_signal, search_ranges=search_indices, slope_direction="positive", traversal_direction="right_to_left", reversal_criterion="all"
    )


    ###########################################
    # Detect b-waves (trough of the APG signal)
    ###########################################

    # Find zero-crossing points for b-wave detection
    search_start = a_wave_locations + 1
    search_end = np.append(search_start[1:], len(apg_signal))
    zero_crossings = zero_crossings_detect(
        apg_signal,
        segment_starts=search_start,
        segment_ends=search_end,
        crossing_type="negative",
        scan_direction="left_to_right",
        selection_mode="first",
    )


    zero_cross_indices = np.array([x[0] for x in zero_crossings])

    # Detect slope reversal points (from zero-crossing) for b-wave locations
    search_start_b = zero_cross_indices
    search_end_b = np.append(zero_cross_indices[1:], len(apg_signal) - 1)
    b_wave_locations = identify_slope_reversals(
        apg_signal,
        slope_direction="positive",
        traversal_direction="left_to_right",
        reversal_criterion="first",
        segment_starts=search_start_b,
        segment_ends=search_end_b,
    )


    b_wave_peaks = apg_signal[b_wave_locations]

    ###########################################
    # Detect e-waves (end of APG signal's cycle)
    ###########################################

    # Generate search indices for e-wave detection based on z-waves
    search_indices_e = []
    for k in range(len(vpg_z_waves)):
        search_indices_e.append(np.arange(vpg_y_waves[k] - 5, vpg_z_waves[k] + 6))

    # Detect slope reversal points from z to y for e-waves in the APG signal
    e_wave_locations = identify_slope_reversals(
        apg_signal, slope_direction="positive", traversal_direction="right_to_left", reversal_criterion="max", search_ranges=search_indices_e
    )


    e_wave_peaks = apg_signal[e_wave_locations]

    ###########################################
    # Detect c-waves and d-waves (intermediate waves)
    ###########################################

    # Search for slope reversals (from e to b) to detect c and d waves
    c_wave_locations = []
    c_wave_peaks = []
    d_wave_locations = []
    d_wave_peaks = []

    for h in range(len(e_wave_locations)):
        segment = apg_signal[b_wave_locations[h] + 1 : e_wave_locations[h] - 1]
        slope = np.diff(segment)

        p = len(slope)
        local_locs = []
        local_peaks = []

        while p > 1:
            if (slope[p - 1] < 0 and slope[p - 2] >= 0) or (slope[p - 1] >= 0 and slope[p - 2] < 0):
                loc = b_wave_locations[h] + p - 1
                peak = apg_signal[loc]
                local_peaks.append(peak)
                local_locs.append(loc)

            p -= 1

        if len(local_locs) >= 2:
            # First detected peak is d-wave, second is c-wave
            d_wave_locations.append(local_locs[0])
            d_wave_peaks.append(local_peaks[0])

            c_wave_locations.append(local_locs[1])
            c_wave_peaks.append(local_peaks[1])

        else:
            # If no valid slope reversal is found, modify the segment between b and e waves
            a1 = b_wave_locations[h]
            b1 = b_wave_peaks[h]
            a2 = e_wave_locations[h]
            b2 = e_wave_peaks[h]

            slope = (b2 - b1) / (a2 - a1)
            modified_segment = apg_signal[a1:a2]

            modified_segment = np.empty(len(range(a1, a2)))

            for i, k in enumerate(range(a1, a2)):
                modified_segment[i] = apg_signal[k] - slope * i

            # Identify the peak in the modified segment
            ind = np.argmax(modified_segment)
            max_index = ind + a1

            # Detect c and d waves based on the modified segment
            segment = modified_segment[ind:]
            slope = np.diff(segment)

            local_locs2 = []
            local_peaks2 = []

            for s in range(len(slope) - 1):
                if (slope[s] < 0 and slope[s + 1] >= 0) or (slope[s] >= 0 and slope[s + 1] < 0):
                    loc = max_index + s
                    peak = apg_signal[loc]
                    local_peaks2.append(peak)
                    local_locs2.append(loc)

            if len(local_locs2) >= 2:
                c_wave_locations.append(local_locs2[0])
                c_wave_peaks.append(local_peaks2[0])

                d_wave_locations.append(local_locs2[1])
                d_wave_peaks.append(local_peaks2[1])

            else:
                # If both attempts fail, consider overlapping c, d, and e waves
                c_wave_locations.append(e_wave_locations[h])
                c_wave_peaks.append(e_wave_peaks[h])

                d_wave_locations.append(e_wave_locations[h])
                d_wave_peaks.append(e_wave_peaks[h])

    # Store the fiducial locations in the output dictionary
    fiducials["a_waves"] = np.asarray(a_wave_locations)
    fiducials["b_waves"] = np.asarray(b_wave_locations)
    fiducials["e_waves"] = np.asarray(e_wave_locations)
    fiducials["c_waves"] = np.asarray(c_wave_locations)
    fiducials["d_waves"] = np.asarray(d_wave_locations)

    return fiducials



def ppg_delineate(
    ppg_signal,
    vpg_signal,
    vpg_fiducials,
    apg_signal,
    apg_fiducials,
    PPG_peaks, 
    PPG_onsets,
    sampling_rate
):
    """Detects fiducial points in the PPG signal.

    Reference:
    Abhishek Chakraborty, Deboleena Sadhukhan & Madhuchhanda Mitra (2019): 
    An Automated Algorithm to Extract Time Plane Features From the PPG Signal and
    its Derivatives for Personal Health Monitoring Applications, IETE Journal of Research, 
    DOI: 10.1080/03772063.2019.1604178

    Args:
        ppg_signal (ArrayLike): The PPG signal.
        vpg_signal (ArrayLike): The VPG signal.
        vpg_fiducials (dict): Dictionary of fiducial points from the VPG signal.
        apg_signal (ArrayLike): The APG signal.
        apg_fiducials (dict): Dictionary of fiducial points from the APG signal.
        sampling_rate (float): Sampling rate of the PPG signal (in Hz).
        onset_locations (ArrayLike, optional): Locations of PPG signal onsets, if available. Defaults to None.

    Returns:
        dict: Dictionary containing the detected fiducial locations (O, S, N, D waves).
    """
    
    # Extract fiducials from VPG and APG signals
    vpg_w_waves = vpg_fiducials["w_waves"]
    apg_e_waves = apg_fiducials["e_waves"]

    fiducial_points = {}

    # ###########################################
    # # Detect S and O-waves (pulse onsets)
    # ###########################################
    systolic_peak_locations = PPG_peaks.copy()
    trough_locations = PPG_onsets.copy()

    ###########################################
    # Detect N-waves (dicrotic notches)
    ###########################################

    # The N-waves are located at the e-wave locations from the APG signal
    dicrotic_notch_locations = apg_e_waves

    ###########################################
    # Detect D-waves (diastolic peaks)
    ###########################################

    # Search for the slope reversal point (from e-wave in the APG signal)
    search_start_D = apg_e_waves + 1
    search_end_D = np.append(apg_e_waves[1:] - 1, len(apg_signal) - 1)

    diastolic_peak_locations = identify_slope_reversals(
        apg_signal,
        slope_direction="positive",
        traversal_direction="left_to_right",
        reversal_criterion="first",
        segment_starts=search_start_D,
        segment_ends=search_end_D,
    )


    # Store detected fiducials in the output dictionary
    fiducial_points["O_waves"] = np.array(trough_locations)
    fiducial_points["S_waves"] = np.array(systolic_peak_locations)
    fiducial_points["N_waves"] = np.array(dicrotic_notch_locations)
    fiducial_points["D_waves"] = np.array(diastolic_peak_locations)

    return fiducial_points










#### ------ ECG Features ------------

def Atrial_VEntricular_Phase(self, ecg, rpeaks, waves, F_s):  
    # Compute the cardiac phase for both atrial and ventricular systole phases
    #ecg: ECG signal
    #rpeaks: R-peak locations
    #waves: Delineated waves
    #F_s: Sampling frequency    
    cardiac_phase = nk.ecg_phase(ecg_cleaned=ecg, rpeaks=rpeaks,
                           delineate_info=waves, sampling_rate=F_s)
    #cardiac_phase: Compute cardiac phase (for both atrial and ventricular),... 
    #...labelled as 1 for systole and 0 for diastole.
    #Atrial Systole: Ppeak to Rpeak, 
    #Atrial Diastole: Rpeak to Ppeak
    #Ventricular Systole: Rpeak to Toffset, 
    #Ventricular Diastole: Toffset to Rpeak
    
    # Compute Atrial Systole Phase
    dAtr_phase_flag = cardiac_phase.ECG_Phase_Atrial.diff()
    Atr_ph_on = np.where(dAtr_phase_flag==1)[0]
    Atr_ph_off = np.where(dAtr_phase_flag==-1)[0]

    if Atr_ph_on[0]>Atr_ph_off[0]: Atr_ph_off = Atr_ph_off[1:]
    LAmin = min([len(Atr_ph_on), len(Atr_ph_off)])  
    Atr_ph_on = Atr_ph_on[:LAmin]
    Atr_ph_off = Atr_ph_off[:LAmin]
    AtrialSystole_phase = 1000*((Atr_ph_off-Atr_ph_on)/F_s) #in ms

    # Compute Ventricular Systole Phase
    dVent_phase_flag = cardiac_phase.ECG_Phase_Ventricular.diff()
    Vent_ph_on = np.where(dVent_phase_flag==1)[0]
    Vent_ph_off = np.where(dVent_phase_flag==-1)[0]

    if Vent_ph_on[0]>Vent_ph_off[0]: Vent_ph_off = Vent_ph_off[1:]
    LVmin = min([len(Vent_ph_on), len(Vent_ph_off)])  
    Vent_ph_on = Vent_ph_on[:LVmin]
    Vent_ph_off = Vent_ph_off[:LVmin]
    VentricularSystole_phase = 1000*((Vent_ph_off-Vent_ph_on)/F_s) #in ms
    return AtrialSystole_phase, VentricularSystole_phase



### SQI functions
# Build a Template Beat around cardiac-peaks (like R-peaks in ECG, and Systolic peaks in PPG)
def extract_beats(self, signal, peaks, fs, window_ms=500):  # 250ms before and after the R-peaks
    half_win = int((window_ms / 1000 / 2) * fs)
    beats = []
    
    for i, r in enumerate(peaks):
        # Normal case: the beat fits within the signal bounds
        if r - half_win >= 0 and r + half_win < len(signal):
            beat = signal[r - half_win : r + half_win]
            beats.append(beat)
        # Edge case: the first beat
        elif r - half_win < 0 and len(peaks) > 1 and r + half_win < len(signal):
            # Take the second beat as the first beat (copying the second beat)
            beat = signal[peaks[1] - half_win : peaks[1] + half_win]  # Adjust to get second beat
            beats.append(beat)
        # Edge case: the last beat
        elif r - half_win >= 0 and len(peaks) > 1 and r + half_win >= len(signal):
            # Take the second last beat as the last beat (copying the second last beat)
            beat = signal[peaks[-2] - half_win : peaks[-2] + half_win]  # Adjust to get second last beat
            beats.append(beat)
    return np.array(beats)

    
def build_template(self, beats):
    return np.mean(beats, axis=0)



def compute_correlation_scores(self, beats, template):
    scores = []
    for beat in beats:
        corr = np.corrcoef(beat, template)[0, 1]
        scores.append(corr)
    return np.array(scores)

#SQI function  ================
def assess_quality(self, signal, fs, peaks):
    beats = self.extract_beats(signal, peaks, fs)
    template = self.build_template(beats)
    beatSQI_scores = self.compute_correlation_scores(beats, template)
    
    mean_corr = np.mean(beatSQI_scores)
    low_corr_ratio = np.sum(beatSQI_scores < 0.8) / len(beatSQI_scores) 
    
    '''
    Discard/Accept the beats based on the following SQI score:
        beatSQI_scores (Closer to 1 = better)
    Discard/Accept the whole signal based on the following SQI scores:
        Mean Correlation Score	(Closer to 1 = more consistent beat morphology)
        Low Correlation Fraction (Higher = more distortion, likely due to motion)
    '''
    return beatSQI_scores, mean_corr, low_corr_ratio

# Discard false beats  ======================
def beat_SQI_decision(self, beatSQI_scores, peaks, fs, corr_thresh=0.70, hr_min=30, hr_max=200):
    """
    Filters out false or noisy R-peaks using beat SQI correlation scores and HR-based constraints.
    
    inputs:
    - beatSQI_scores: array-like of correlation scores for each beat
    - peaks: array of R-peak indices (sample positions)
    - fs: sampling frequency in Hz
    - corr_thresh: minimum acceptable correlation coefficient (default = 0.70)
    - hr_min, hr_max: HR bounds (default = 30–200 bpm)

    output:
    - peaks_final: filtered R-peaks
    """
    peaks = np.array(peaks)
    beatSQI_scores = np.array(beatSQI_scores)
    cardiac_intervals = np.diff(peaks) * 1000 / fs #in ms
    
    # Instantaneous HR from RR (bpm)
    HR = 60 * 1000 / cardiac_intervals 
    hr_valid = (HR >= hr_min) & (HR <= hr_max)
    hr_valid_extended = np.append(hr_valid[0], hr_valid)

    ## Discard both beats if abnormal RR: P(n-1) and P(n)
    #hr_valid_extended[np.where(hr_valid_extended == 0)[0] - 1] = 0

    morph_valid = beatSQI_scores > corr_thresh # Correlation filter

    # Combine both filters
    valid_beats_mask = morph_valid & (hr_valid_extended == 1)
    peaks_final = pd.Series(peaks)[valid_beats_mask]

    return peaks_final.values

# Final SQI function to be used in the main function
def cardiac_SQI_beats(self, signal, Fs, peaks):
    """
    Filters out false or noisy cardiac-peaks using beat SQI correlation scores and HR-based constraints.
    
    inputs:
    - signal: cardiac signal
    - Fs: sampling frequency in Hz
    - peaks: array of cardiac-peak indices (sample positions)

    output:
    - peaks_final: filtered cardiac peaks
    """
    peaks = np.array(peaks)
    beatSQI_scores, mean_corr, low_corr_ratio = self.assess_quality(signal, Fs, peaks)
    peaks_final = self.beat_SQI_decision(beatSQI_scores, peaks, Fs)
    return peaks_final, beatSQI_scores, mean_corr, low_corr_ratio



def ECG_features_delineate(self, ecg_signal, Fs, info):
    features_dict = {}
    # R_flag, info = nk.ecg_peaks(ecg_signal, sampling_rate=Fs, method="neurokit") 
    Rpks = info['ECG_R_Peaks']
    beatSQI_scores, mean_corr, low_corr_ratio = self.assess_quality(ecg_signal, Fs, Rpks)
    Rtrue = self.beat_SQI_decision(beatSQI_scores, Rpks, Fs)
    _, waves = nk.ecg_delineate(ecg_signal, Rtrue, sampling_rate=Fs) 
    atrialSys_phase_int, ventSys_phase_int = self.Atrial_VEntricular_Phase(ecg_signal, info, waves, Fs)
    
    features_dict['atrialSys_phase'] = atrialSys_phase_int
    features_dict['ventSys_phase'] = ventSys_phase_int

    # interval features
    if "ECG_P_Onsets" in waves and "ECG_P_Offsets" in waves:
        p_duration = np.array(waves["ECG_P_Offsets"]) - np.array(waves["ECG_P_Onsets"])
        features_dict["P_duration"] = p_duration
    else:
        features_dict["P_duration"] = np.nan

    if "ECG_R_Onsets" in waves and "ECG_R_Offsets" in waves:
        qrs_duration = np.array(waves["ECG_R_Offsets"]) - np.array(waves["ECG_R_Onsets"])
        features_dict["QRS_duration"] = qrs_duration
    else:
        features_dict["QRS_duration"] = np.nan

    if "ECG_T_Onsets" in waves and "ECG_T_Offsets" in waves:
        t_duration = np.array(waves["ECG_T_Offsets"]) - np.array(waves["ECG_T_Onsets"])
        features_dict["T_duration"] = t_duration
    else:
        features_dict["T_duration"] = np.nan


    pr_intervals_ms = []
    pr_segment_ms = []
    for r in Rtrue:
        # Find latest P-onset before the R-peak
        pon_candidates = np.array(waves['ECG_P_Onsets'])[waves['ECG_P_Onsets'] < r]
        poff_candidates = np.array(waves['ECG_P_Offsets'])[waves['ECG_P_Offsets'] < r]
        ron_candidates = np.array(waves['ECG_R_Onsets'])[waves['ECG_R_Onsets'] < r]

        if len(pon_candidates) == 0 or len(ron_candidates) == 0:
            pr_intervals_ms.append(np.nan)
            continue
        if len(poff_candidates) == 0 or len(ron_candidates) == 0:
            pr_segment_ms.append(np.nan)
            continue
        p_onset = pon_candidates[-1]
        r_onset = ron_candidates[-1]
        p_offset = poff_candidates[-1]
        if r_onset < p_onset:  # Invalid if R_on comes before P onset
            pr_intervals_ms.append(np.nan)
            continue
        if r_onset < p_offset:  # Invalid if R_on comes before P onset
            pr_segment_ms.append(np.nan)
            continue
        pr_interval = (r_onset - p_onset) / Fs * 1000  # in ms
        pr_segment = (r_onset - p_offset) / Fs * 1000  # in ms
        pr_intervals_ms.append(pr_interval)
        pr_segment_ms.append(pr_segment)  
    if np.nanmean(pr_intervals_ms)<400: features_dict["pr_interval_ms"] = pr_intervals_ms #must be <400 ms
    else: features_dict["pr_interval_ms"] = np.nan
    if np.nanmean(pr_segment_ms)<300: features_dict["pr_segment_ms"] = pr_segment_ms #must be <300 ms
    else: features_dict["pr_segment_ms"] = np.nan


    st_intervals_ms = []
    st_segment_ms = []
    for r in Rtrue:
        # Find latest T-onset, T-offset after the R-peak
        roff_candidates = np.array(waves['ECG_R_Offsets'])[waves['ECG_R_Offsets'] > r]
        toff_candidates = np.array(waves['ECG_T_Offsets'])[waves['ECG_T_Offsets'] > r]
        ton_candidates = np.array(waves['ECG_T_Onsets'])[waves['ECG_T_Onsets'] > r]
        if len(roff_candidates) == 0 or len(ton_candidates) == 0:
            st_segment_ms.append(np.nan)
            continue
        if len(roff_candidates) == 0 or len(toff_candidates) == 0:
            st_intervals_ms.append(np.nan)
            continue
        r_offset = roff_candidates[0]
        t_offset = toff_candidates[0]
        t_onset = ton_candidates[0]
        if r_offset > t_offset:  
            st_intervals_ms.append(np.nan)
            continue
        if r_offset > t_onset:  
            st_segment_ms.append(np.nan)
            continue
        st_interval = (t_offset - r_offset) / Fs * 1000  # in ms
        st_segment = (t_onset - r_offset) / Fs * 1000  # in ms
        st_intervals_ms.append(st_interval)
        st_segment_ms.append(st_segment) 
    if np.nanmean(st_intervals_ms)<700: features_dict["st_interval_ms"] = st_intervals_ms
    else: features_dict["st_interval_ms"] = np.nan
    if np.nanmean(st_segment_ms)<400: features_dict["st_segment_ms"] = st_segment_ms
    else: features_dict["st_segment_ms"] = np.nan


    qt_intervals_ms = []
    qtc_intervals_ms = []
    for i, r in enumerate(Rtrue):
        # Get the last R-onset before this R-peak, and latest T-offset after the R-peak
        ron_candidates = np.array(waves['ECG_R_Onsets'])[waves['ECG_R_Onsets'] < r]
        toff_candidates = np.array(waves['ECG_T_Offsets'])[waves['ECG_T_Offsets'] > r]
        if len(ron_candidates) == 0 or len(toff_candidates) == 0 or i == 0:
            qt_intervals_ms.append(np.nan)
            qtc_intervals_ms.append(np.nan)
            continue
        r_onset = ron_candidates[-1]
        t_offset = toff_candidates[0]
        if r_onset > t_offset:
            qt_intervals_ms.append(np.nan)
            qtc_intervals_ms.append(np.nan)
            continue
        # QT interval in ms
        qt_interval_sec = (t_offset - r_onset) / Fs
        qt_intervals_ms.append(qt_interval_sec * 1000)
        # RR interval in seconds
        rr_interval_sec = (r - Rtrue[i - 1]) / Fs
        if rr_interval_sec == 0:
            qtc_intervals_ms.append(np.nan)
        else:
            # Fridericia's correction
            qtc = qt_interval_sec / (rr_interval_sec ** (1/3)) #in sec
            qtc_intervals_ms.append(qtc * 1000)

    # Save to feature dictionary
    features_dict["qt_interval_ms"] = qt_intervals_ms if np.nanmean(qt_intervals_ms) < 900 else np.nan
    features_dict["qtc_interval_ms"] = qtc_intervals_ms if np.nanmean(qtc_intervals_ms) < 900 else np.nan
    return features_dict, Rtrue



def signal_center_frequency(self, signal, sampling_rate, show=False):
    ## Center frequency (CF), also known as center of gravity
    # welch frequency * power spectral density / summation of power spectral density
    welch = nk.signal_psd(signal, method="welch", sampling_rate=sampling_rate, min_frequency=0, max_frequency=sampling_rate/2, show=show)
    f = welch['Frequency']
    psd = welch['Power']
    numerator = np.sum(f * psd)
    denominator = np.sum(psd)
    center_frequency = numerator / denominator if denominator != 0 else 0
    return center_frequency









##FUNCTION: Calculation of pulse arrival time
def pulse_arrival_time(self, ecg_Rpeaks, PPGonsets, F_s, cardiac_int_ms):
    ecg_Rpeaks = np.array(ecg_Rpeaks)
    PPGonsets = np.array(PPGonsets)
    #ecg_R to ppg_foot: pulse arrival time
    if len(ecg_Rpeaks)==0 or len(PPGonsets)==0:
        pat = np.nan
        dpat = np.nan
    else:
        pat = []
        for ecg_r in ecg_Rpeaks:
            PAT_all = (PPGonsets - ecg_r) * 1000 /F_s
            PAT_all = PAT_all[PAT_all>0]
            if PAT_all.size != 0:
                mPAT_all = PAT_all.min()
                if mPAT_all < (cardiac_int_ms*0.85):
                    pat.append(mPAT_all)
        if len(pat) != 0: 
            dpat = pd.Series(pat).diff()
        else: 
            dpat = np.nan
    return pat, dpat





###FUNCTION for PPG class identification : Reference: doi: 10.1088/1361-6579/ad33a2
''' The fiducial point dn can be classified into four classes: 
    Class 1, in which the dn is an incisura, 
    Class 2, in which there is a horizontal line at the dn, 
    Class 3, in which there is a change in gradient on the downslope, and 
    Class 4, in which there is no clear evidence of the dn.
'''
def PPG_wave_class(self, ppg_signal, ppg_fiducials):
    class_waveform = "Unknown"
    try: 
        S_waves, N_waves, O_waves, D_waves = \
            [ppg_fiducials.get(key) for key in ('S_waves', 'N_waves', 'O_waves', 'D_waves')]
        
        if len(D_waves) != 0:
            # Extract fiducial points from dictionaries

            med_aDN = np.median(ppg_signal[N_waves])
            med_aDP = np.median(ppg_signal[D_waves])
            
            per_change = np.abs(100 * (med_aDP - med_aDN) / med_aDN)
            
            # Class 1:     
            if (med_aDP > med_aDN) and per_change > 10:
                return "Class1"
            
            # Class 2:     
            elif per_change < 10:
                return "Class2"
            
            # Class 3:
            elif (med_aDP < med_aDN) and per_change > 10:
                return "Class3"
            
            # Class 4
            elif np.isnan(med_aDN) is True:
                return "Class4"
            
            else:
                return class_waveform
        else: class_waveform

    except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return class_waveform 




def pulse_width(self, locs_onsets, F_s, cardiac_int_ms):
    if len(locs_onsets)>1:
        pulse_width = np.diff(locs_onsets)*1000/F_s # in ms
        PW_ms = pulse_width[np.where(pulse_width < (cardiac_int_ms+150))] #putting an HR-based constraint with a relaxation of 150 ms to avoid outliers
    else: PW_ms = np.nan
    return PW_ms



def pulse_rate_ppg(self, loc_peaks, F_s, cardiac_int_ms):
    if len(loc_peaks)>1:
        pulse_int = np.diff(loc_peaks)*1000/F_s # in seconds
        pulse_interval_ms = pulse_int[np.where(pulse_int < (cardiac_int_ms+150))] #putting an HR-based constraint with a relaxation of 150 ms to avoid outliers
        pulse_rate = 1000*60/pulse_interval_ms
    else: 
        pulse_interval_ms = np.nan
        pulse_rate = np.nan
    return pulse_interval_ms, pulse_rate
    


def ppg_peak_amp_difference(self, ppg_signal, locs_peaks):
    #Amplitude difference between the consecutive systolic peaks
    #locs_peaks: PPG's systolic peak locations
    #ppg_signal: PPG signal
    if len(locs_peaks)>1:
        Pamp_diff_ppg = np.diff(ppg_signal[locs_peaks])
        Pamp_diff_ppg = Pamp_diff_ppg/ppg_signal[locs_peaks[0]] #normalization
    else:
        Pamp_diff_ppg = np.nan
    return Pamp_diff_ppg

    


def maximum_upslope_ppg(self, vpg_signal, w_waves):
    #maximum first derivative of the PPG peak amplitude, use vpg signal
    if len(w_waves) == 0: 
        max_upslope = np.nan
    else: max_upslope = vpg_signal[w_waves]
    return max_upslope






def ppg_systole_diastole_delineation(self, ppg_signal, PPGonset_samples, PPGpeak_samples, DP_samples, DN_samples, F_s, cardiac_int_ms):
    PPGonset = np.array(PPGonset_samples)
    onset = PPGonset[:-1]
    offset = PPGonset[1:]
    
    SP = np.array(PPGpeak_samples)
    DP = np.array(DP_samples)
    DN = np.array(DN_samples)
    

    '''
    Output: 
        Delta_T_sd_ms: Systole to diastole duration (ms)
        RI: Reflection index = diastole_amp / systole_amp w.r.t. onset amplitude
        SI: Stiffness Index, the ratio of the systolic peak amplitude to the systole to diastole Time duration
        Delta_A_dn_dp: DN to DP amplitude difference
        T_dn_dp_ms: DN to DP time interval
        
        Ref. for SI: 
        [A] Millasseau S C, Kelly R P, Ritter J M and Chowienczyk P J 2002 Determination of age-related increases in large artery stiffness by digital pulse contour analysis Clin. Sci. 103 371–7.
        [B] Márton Á Goda et al pyPPG: a Python toolbox for comprehensive photoplethysmography signal analysis 2024 Physiol. Meas. 45 045001
    '''
    #Return None when no DP exists (or detected), and/or existing DP-peaks don't come under onset-offset regions 
    if len(DP) == 0: #Case 1: when no DP exists (or detected)
        Delta_T_sd_ms = np.nan #Systole to diastole duration
        RI = np.nan #Reflection index
        Delta_A_dn_dp = np.nan #Amp. difference of DP and DN
        T_dn_dp_ms = np.nan #DN to DP time
        SI = np.nan # Stiffness Index, the ratio of the systolic peak amplitude to the systole to diastole Time duration.
        
    else:
        ##step 1: find the correct pairings of onsets, SP, DN, DP and offsets in each ppg pulse
        SP_sync = []
        On_sync = []
        Off_sync = []
        DP_sync = []
        DN_sync = []
        for k in range(len(onset)):
            cond1 = SP > onset[k]
            cond2 = SP < offset[k]
            SP_all = pd.Series(SP[np.where(cond1 & cond2)]) #SP_m must be a single scaler
            
            if len(SP_all) > 0: 
                SP_m = SP_all[0]  
            
                cond1 = DP > SP_m
                cond2 = DP < offset[k]
                DP_all = pd.Series(DP[np.where(cond1 & cond2)])
                
                if len(DN) != 0:
                    cond0 = DN > onset[k]
                    cond1 = DN > SP_m
                    cond2 = DN < offset[k]
                    DN_all = pd.Series(DN[np.where(cond0 & cond1 & cond2)])
                    if len(DN_all) != 0:
                        DN_sync = np.append(DN_sync, DN_all[0])

                
                if len(DP_all)!=0:
                    SP_sync = np.append(SP_sync, SP_m)
                    On_sync = np.append(On_sync, onset[k])
                    Off_sync = np.append(Off_sync, offset[k])
                    DP_sync = np.append(DP_sync, DP_all[0])
                
        #step 2: extract features 
        SP_sync = np.array(SP_sync, dtype=int)
        DP_sync = np.array(DP_sync, dtype=int)
        On_sync = np.array(On_sync, dtype=int)
        Off_sync = np.array(Off_sync, dtype=int)

        
        if len(DP_sync) == 0: #Case 2: existing DP-peaks don't come under onset-offset regions 
            Delta_T_sd_ms = np.nan #Systole to diastole duration
            RI = np.nan #Reflection index
            Delta_A_dn_dp = np.nan #Amp. difference of DP and DN
            T_dn_dp_ms = np.nan #DN to DP time
            SI = np.nan #SI
        else:
            Del_T = (DP_sync - SP_sync)*1000/F_s
            Delta_T_sd_ms = Del_T[np.where(Del_T < cardiac_int_ms*0.60)]
            h1 = ppg_signal[DP_sync] - ppg_signal[On_sync[0]]
            h2 = ppg_signal[SP_sync] - ppg_signal[On_sync[0]]
            RI = h1/h2
            
            # Stiffness Index, the ratio of the systolic peak amplitude to the systole to diastole Time duration
            SI = 1000*h2/Del_T
            
            if len(DN_sync) != 0:
                DN_sync = np.array(DN_sync, dtype=int)
                minlen_DN_DP = min(len(DN_sync), len(DP_sync))
                DN_sync = DN_sync[:minlen_DN_DP]
                DP_sync = DP_sync[:minlen_DN_DP] 
                Delta_A_dn_dp = (ppg_signal[DP_sync] - ppg_signal[DN_sync]) / ppg_signal[On_sync[0]] #Nor. Amp. difference of DP and DN
                T_dn_dp_ms = (DP_sync - DN_sync)*1000/F_s
            else: 
                Delta_A_dn_dp = np.nan
                T_dn_dp_ms = np.nan

    return Delta_T_sd_ms, RI, Delta_A_dn_dp, T_dn_dp_ms, SI
        






def augmentation_index_ppg(self, ppg_signal, PPGpeak_samples, PPGonset_samples, apg_fiducials): 
    #AI = [x(p2)–x(onset)]/[x(p1)–x(onset)], where p1: systolic peak-1, p2: systolic peak-2, p1, p2: early and late systolic peaks
    # p1 is PPGpeak, and p2 can be extracted from APG fiducials (c_waves)
    c_waves = apg_fiducials.get('c_waves') 
    
    if len(c_waves) == 0: AI_final = np.nan
    else:
        PPGpeak = PPGpeak_samples.to_numpy()
        PPGonset = PPGonset_samples.to_numpy()
        
        PPGpeak0 = PPGpeak[0]
        PPGonset0 = PPGonset[0]
        if PPGpeak0 > PPGonset0:
            PPGonset = PPGonset[1:]
        min_length = min(len(PPGpeak), len(PPGonset))
        PPGpeak_new = PPGpeak[:min_length]
        PPGonset_new = PPGonset[:min_length]
        
        AI_final = []
        for sys, onset in zip(PPGpeak_new, PPGonset_new):
            #Which c wave is in between sys and onset for each pulsatile cycle
            cond1 = c_waves > sys 
            cond2 = c_waves < onset
            p2 = pd.Series(c_waves[np.where(cond1 & cond2)])
            if p2.size != 0: 
                if len(p2) > 1: p2 = p2[0]
                p2_ppg = ppg_signal[p2] - ppg_signal[PPGonset_new[0]]
                p1_ppg = ppg_signal[sys] - ppg_signal[PPGonset_new[0]]
                AI = p2_ppg/p1_ppg
                AI_final = np.append(AI_final, AI)
    return AI_final        
    




def statistical_measure_ppg(self, ppg_signal, PPGonsets_samples, F_s, cardiac_int_ms): 
    # Statistical measures for each PPG pulsatile wave
    locs_onsets = np.array(PPGonsets_samples) 
    onset = locs_onsets[:-1]
    offset = locs_onsets[1:]
    
    mean, median, variance, skewness, kurt_values, std, entropy_val, center_freq, energy = [], [], [], [], [], [], [], [], []
    for i in range(len(onset)):
        Tpulse_ms = (offset[i] - onset[i])*1000/F_s
        if Tpulse_ms < (cardiac_int_ms+180):
            start_pulse_index = locs_onsets[i]
            end_pulse_index = locs_onsets[i+1]
            wave_cycle = ppg_signal[start_pulse_index : end_pulse_index]
            wave_cycle = wave_cycle-np.min(wave_cycle) ### Making the pulse positive on zero-line 
            mean_values = np.mean(wave_cycle)
            median_values = np.median(wave_cycle)
            variance_values = np.var(wave_cycle)
            std_values = np.std(wave_cycle)
            skew_value = skew(wave_cycle)
            kurt_value = kurtosis(wave_cycle)
            energy_val = np.sum(wave_cycle**2)

            #shannon entropy = sum(-pi*log2(pi))
            histogram, _ = np.histogram(wave_cycle, bins=30, density = True)
            hist_normalized = histogram/np.sum(histogram)
            base = 2
            entropy = -np.sum(hist_normalized * np.log(hist_normalized+np.finfo(float).eps)) / np.log(base)  
            try: 
                center_frequency = self.signal_center_frequency(wave_cycle, F_s, show=False)
            except: 
                center_frequency=np.nan
            mean.append(mean_values)
            median.append(median_values)
            variance.append(variance_values)
            std.append(std_values)
            skewness.append(skew_value)
            kurt_values.append(kurt_value)
            entropy_val.append(entropy)
            center_freq.append(center_frequency)
            energy.append(energy_val)
        
            
    return mean, median, variance, skewness, kurt_values, std, entropy_val, center_freq, energy




############# PPG AREA-BASED FEATURES ##################

def pulse_area(self, ppg_signal, PPGonset_samples, F_s, cardiac_int_ms, AC_amp_med): #same cycle
    PPGonset = np.array(PPGonset_samples)  
    onset = PPGonset[:-1]
    offset = PPGonset[1:]
    
    T_pulse = (offset - onset)*1000/F_s  #in ms
    #putting an HR-based constraint to avoid outliers
    
    pulse_areas = []
    for i in range(len(onset)):
        if T_pulse[i] < (cardiac_int_ms*1.75):
            wave_cycle = ppg_signal[onset[i] : offset[i]]
            wave_cycle = wave_cycle-np.min(wave_cycle) ### Making the pulse positive on zero-line 
            area = np.trapz(wave_cycle)
            pulse_areas.append(area)
            # plt.figure()
            # plt.plot(wave_cycle)
            # plt.title(f'area: {area}') 
            
    ### Normalization
    T_pulse_samples = T_pulse*F_s/1000
    mT_pulse_samples = np.median(T_pulse_samples)
    mAC_amp = np.median(AC_amp_med)
    
    pulse_areas_N = pulse_areas / (mT_pulse_samples * mAC_amp)
    return pulse_areas_N
            
    
    
def systolic_diastolic_area_time(self, ppg_signal, PPGonset_samples, DN_samples, F_s, cardiac_int_ms, AC_amp_med): #same cycle
    PPGonset = np.array(PPGonset_samples)
    onset = PPGonset[:-1]
    offset = PPGonset[1:]
    
    DN = np.array(DN_samples)
    
    #Return None when no DN exists (or detected), and/or existing DNs don't come under onset-offset regions 
    if len(DN) == 0: # Case 1: when no DN exists (or detected)
        systolic_areas_N = np.nan
        diastolic_areas_N = np.nan
        IPA = np.nan
        T_sys_f = np.nan
        T_dias_f = np.nan
        DN_exists = False
        A_dn = np.nan
    else:
        ##step 1: find the correct pairings of onsets, DN, offsets in each ppg pulse
        DN_exists = True
        DN_sync = []
        On_sync = []
        Off_sync = []
        for k in range(len(onset)):
            cond1 = DN > onset[k]
            cond2 = DN < offset[k]
            DN_all = pd.Series(DN[np.where(cond1 & cond2)])
            if len(DN_all) != 0:
                DN_sync = np.append(DN_sync, DN_all[0])
                On_sync = np.append(On_sync, onset[k])
                Off_sync = np.append(Off_sync, offset[k]) 
        ##step 2: (A) systole and diastole time durations (putting an HR-based constraint to avoid outliers)
        ##step 2: (B) Calculation of areas and their normalization by (pulse_width x AC amplitude) to make generalizable
        
        if len(DN_sync) == 0: # Case 2: when existing DNs don't come under onset-offset regions 
            systolic_areas_N = np.nan
            diastolic_areas_N = np.nan
            IPA = np.nan
            T_sys_f = np.nan
            T_dias_f = np.nan
            DN_exists = False
            A_dn = np.nan
            
        else:    
            T_sys = (DN_sync - On_sync)*1000/F_s  #in ms
            T_dias = (Off_sync - DN_sync)*1000/F_s  #in ms
            
            
            DN_sync = np.array(DN_sync, dtype=int)
            On_sync = np.array(On_sync, dtype=int)
            Off_sync = np.array(Off_sync, dtype=int)
            
            A_dn = ppg_signal[DN_sync] - ppg_signal[On_sync[0]]    ##PPG's DN amplitude wrt onset
            
                
            
            systolic_areas = []
            diastolic_areas = []
            T_sys_f = []
            T_dias_f = []
            for i in range(len(On_sync)):
                if T_sys[i] < (cardiac_int_ms*0.80) and T_dias[i] < (cardiac_int_ms*0.80): #80% of cardiac interval
                    start1 = On_sync[i]
                    end1 = DN_sync[i]
                    wave1 = ppg_signal[start1:end1]
                    wave1 = wave1-np.min(wave1) ### Making the wave positive on zero-line 
                    area_sys = np.trapz(wave1)
                    start2 = DN_sync[i]
                    end2 = Off_sync[i]
                    wave2 = ppg_signal[start2:end2]
                    wave2 = wave2-np.min(wave2) ### Making the wave positive on zero-line
                    area_dias = np.trapz(wave2)
                    
                    systolic_areas.append(area_sys)
                    diastolic_areas.append(area_dias)
                    T_sys_f.append(T_sys[i])
                    T_dias_f.append(T_dias[i])
                    # fig, ax = plt.subplots(1,3, figsize=(11,3))
                    # ax[0].plot(ppg_signal[start1:end1])
                    # ax[1].plot(ppg_signal[start2:end2])
                    # ax[2].plot(ppg_signal[start1:end2])
                    # ax[1].set_title(f'area_sys, area_dias: {i}=={np.round(area_sys, 3)}=={np.round(area_dias, 3)}') 
            
            ### Normalization
            T_pulse = np.array(T_sys_f) + np.array(T_sys_f)
            T_pulse_samples = T_pulse*F_s/1000
            mT_pulse_samples = np.median(T_pulse_samples)
            mAC_amp = np.median(AC_amp_med)
            
            systolic_areas_N = systolic_areas / (mT_pulse_samples * mAC_amp)
            diastolic_areas_N = diastolic_areas / (mT_pulse_samples * mAC_amp)  
            IPA = np.array(diastolic_areas_N) /  np.array(systolic_areas_N)
    return systolic_areas_N, diastolic_areas_N, IPA, T_sys_f, T_dias_f, DN_exists, A_dn



def AUCow_AUCwo_area_time(self, ppg_signal, PPGonset_samples, VPG_W_samples, F_s, cardiac_int_ms, AC_amp_med): #same cycle
    PPGonset = np.array(PPGonset_samples)
    onset = PPGonset[:-1]
    offset = PPGonset[1:]
    
    W = np.array(VPG_W_samples)
    
    #Return None when no W exists (or detected), and/or existing W-peaks don't come under onset-offset regions 
    if len(W) == 0: #Case 1: when no W exists (or detected)
        AUCow_N = np.nan
        AUCwo_N = np.nan
        T_ow_f = np.nan
        T_wo_f = np.nan
    else:
        ##step 1: find the correct pairings of onsets, W, offsets in each ppg pulse
        W_sync = []
        On_sync = []
        Off_sync = []
        for k in range(len(onset)):
            cond1 = W > onset[k]
            cond2 = W < offset[k]
            W_all = pd.Series(W[np.where(cond1 & cond2)])
            if len(W_all) != 0:
                W_sync = np.append(W_sync, W_all[0])
                On_sync = np.append(On_sync, onset[k])
                Off_sync = np.append(Off_sync, offset[k])
        ##step 2: (A) systole and diastole time durations (putting an HR-based constraint to avoid outliers)
        ##step 2: (B) Calculation of areas and their normalization by (pulse_width x AC amplitude) to make generalizable
        if len(W_sync) == 0: #Case 2: existing W-peaks don't come under onset-offset regions 
            AUCow_N = np.nan
            AUCwo_N = np.nan
            T_ow_f = np.nan
            T_wo_f = np.nan
        else:
            T_ow = (W_sync - On_sync)*1000/F_s  #in ms
            T_wo = (Off_sync - W_sync)*1000/F_s  #in ms
            
            AUCow = []
            AUCwo = []
            T_ow_f = []
            T_wo_f = []
            for i in range(len(On_sync)):
                if T_ow[i] < (cardiac_int_ms*0.50) and T_wo[i] < cardiac_int_ms: #cardiac interval-based constraint
                    start1 = int(On_sync[i])
                    end1 = int(W_sync[i])
                    wave1 = ppg_signal[start1:end1]
                    wave1 = wave1-np.min(wave1) ### Making the wave positive on zero-line
                    area_ow = np.trapz(wave1)
                    start2 = int(W_sync[i])
                    end2 = int(Off_sync[i])
                    wave2 = ppg_signal[start2:end2]
                    wave2 = wave2-np.min(wave2) ### Making the wave positive on zero-line
                    area_wo = np.trapz(wave2)
                    
                    AUCow.append(area_ow)
                    AUCwo.append(area_wo)
                    T_ow_f.append(T_ow[i])
                    T_wo_f.append(T_wo[i])
                    # fig, ax = plt.subplots(1,3, figsize=(11,3))
                    # ax[0].plot(ppg_signal[start1:end1])
                    # ax[1].plot(ppg_signal[start2:end2])
                    # ax[2].plot(ppg_signal[start1:end2])
                    # ax[1].set_title(f'AUCow, AUCwo: {i}=={np.round(area_ow, 3)}=={np.round(area_wo, 3)}') 
            
            ### Normalization
            T_pulse = np.array(T_ow_f) + np.array(T_wo_f)
            T_pulse_samples = T_pulse*F_s/1000
            mT_pulse_samples = np.median(T_pulse_samples)
            mAC_amp = np.median(AC_amp_med)

            AUCow_N = AUCow / (mT_pulse_samples * mAC_amp)
            AUCwo_N = AUCwo / (mT_pulse_samples * mAC_amp)
    return AUCow_N, AUCwo_N, T_ow_f, T_wo_f



def AUCos_AUCso_area_time(self, ppg_signal, PPGonset_samples, PPGpeak_samples, F_s, cardiac_int_ms, vpg_signal): #same cycle
    PPGonset = np.array(PPGonset_samples)
    onset = PPGonset[:-1]
    offset = PPGonset[1:]
    
    SP = np.array(PPGpeak_samples)
    
    #Return None when no SP exists (or detected), and/or existing SP-peaks don't come under onset-offset regions 
    if len(SP) == 0: #Case 1: when no SP exists (or detected)  
        AUCos_N = np.nan
        AUCso_N = np.nan
        T_os_f = np.nan
        T_so_f = np.nan
        T_so_cd = np.nan
        A_AC = np.nan
        A_sp = np.nan
        A_off = np.nan
        mean_slope_os = np.nan
        mean_slope_so = np.nan
        PW_10, PW_25, PW_33, PW_50, PW_66, PW_75 = np.nan*np.ones(6)
            
    else:
        ##step 1: find the correct pairings of onsets, SP, offsets in each ppg pulse
        SP_sync = []
        On_sync = []
        Off_sync = []
        for k in range(len(onset)):
            cond1 = SP > onset[k]
            cond2 = SP < offset[k]
            SP_all = pd.Series(SP[np.where(cond1 & cond2)])
            if len(SP_all) != 0:
                SP_sync = np.append(SP_sync, SP_all[0])
                On_sync = np.append(On_sync, onset[k])
                Off_sync = np.append(Off_sync, offset[k])
        ##step 2: (A) systole and diastole time durations (putting an HR-based constraint to avoid outliers)
        ##step 2: (B) Calculation of areas and their normalization by (pulse_width x AC amplitude) to make generalizable
        if len(SP_sync) == 0: #Case 2: esisting SP-peaks don't come under onset-offset regions
            AUCos_N = np.nan
            AUCso_N = np.nan
            T_os_f = np.nan
            T_so_f = np.nan
            T_so_cd = np.nan
            A_AC = np.nan
            A_sp = np.nan
            A_off = np.nan
            mean_slope_os = np.nan
            mean_slope_so = np.nan
            PW_10, PW_25, PW_33, PW_50, PW_66, PW_75 = np.nan*np.ones(6)
        else:
            T_os = (SP_sync - On_sync)*1000/F_s  #in ms
            T_so = (Off_sync - SP_sync)*1000/F_s  #in ms
            PW = (Off_sync - On_sync)*1000/F_s #in ms
            
            SP_sync = np.array(SP_sync, dtype=int)
            On_sync = np.array(On_sync, dtype=int)
            Off_sync = np.array(Off_sync, dtype=int)
            
            A_sp = ppg_signal[SP_sync] - ppg_signal[On_sync[0]]    ##PPG's SP amplitude wrt onset
            A_off = ppg_signal[Off_sync] - ppg_signal[On_sync[0]]  ##Offset amp. wrt onset 
            
            AUCos = []
            AUCso = []
            T_os_f = [] #Crest time (a.k.a. rise time)
            T_so_f = [] #P2O: peak to offset time 
            T_so_cd = [] #P2O_cd: P2O time corrected
            A_AC = [] #AC amplitude 
            mean_slope_os = []
            mean_slope_so = []
            PW_10 = []
            PW_25 = []
            PW_33 = []
            PW_50 = []
            PW_66 = []
            PW_75 = []
            for i in range(len(On_sync)):
                if T_os[i] < (cardiac_int_ms*0.80) and T_so[i] < (cardiac_int_ms*0.80): #cardiac interval-based constraint
                    start1 = On_sync[i]
                    end1 = SP_sync[i]
                    wave1 = ppg_signal[start1:end1]
                    wave1 = wave1-np.min(wave1) ### Making the wave positive on zero-line
                    area_os = np.trapz(wave1)
                    start2 = SP_sync[i]
                    end2 = Off_sync[i]
                    wave2 = ppg_signal[start2:end2]
                    wave2 = wave2-np.min(wave2) ### Making the wave positive on zero-line
                    area_so = np.trapz(wave2)
                    P2Ocd = T_so[i]/PW[i]
                    AC = ppg_signal[end1] - ppg_signal[start1]
                    
                    m_slope_os = np.mean(vpg_signal[start1:end1])
                    m_slope_so = np.mean(vpg_signal[start2:end2])
                    
                    #pulse_width_at_levelX: the width at X% of the SP Amplitude between the pulse onset and offset.
                    PW_X_10, PW_os_X_10, PW_so_X_10 = self.PW_at_X_level(ppg_signal, On_sync[i],  SP_sync[i], Off_sync[i], On_sync[0], A_sp[i], F_s, X_percent=10)
                    PW_X_25, PW_os_X_25, PW_so_X_25 = self.PW_at_X_level(ppg_signal, On_sync[i],  SP_sync[i], Off_sync[i], On_sync[0], A_sp[i], F_s, X_percent=25)
                    PW_X_33, PW_os_X_33, PW_so_X_33 = self.PW_at_X_level(ppg_signal, On_sync[i],  SP_sync[i], Off_sync[i], On_sync[0], A_sp[i], F_s, X_percent=33)
                    PW_X_50, PW_os_X_50, PW_so_X_50 = self.PW_at_X_level(ppg_signal, On_sync[i],  SP_sync[i], Off_sync[i], On_sync[0], A_sp[i], F_s, X_percent=50)
                    PW_X_66, PW_os_X_66, PW_so_X_66 = self.PW_at_X_level(ppg_signal, On_sync[i],  SP_sync[i], Off_sync[i], On_sync[0], A_sp[i], F_s, X_percent=66)
                    PW_X_75, PW_os_X_75, PW_so_X_75 = self.PW_at_X_level(ppg_signal, On_sync[i],  SP_sync[i], Off_sync[i], On_sync[0], A_sp[i], F_s, X_percent=75)
                            
                    AUCos.append(area_os)
                    AUCso.append(area_so)
                    T_os_f.append(T_os[i])
                    T_so_f.append(T_so[i])
                    T_so_cd.append(P2Ocd)
                    A_AC.append(AC)
                    mean_slope_os.append(m_slope_os)
                    mean_slope_so.append(m_slope_so)
                    PW_10.append(PW_X_10)
                    PW_25.append(PW_X_25)
                    PW_33.append(PW_X_33)
                    PW_50.append(PW_X_50)
                    PW_66.append(PW_X_66)
                    PW_75.append(PW_X_75)
                        
                    # fig, ax = plt.subplots(1,3, figsize=(11,3))
                    # ax[0].plot(ppg_signal[start1:end1])
                    # ax[1].plot(ppg_signal[start2:end2])
                    # ax[2].plot(ppg_signal[start1:end2])
                    # ax[1].set_title(f'AUCos, AUCso: {i}=={np.round(area_os, 3)}=={np.round(area_so, 3)}') 
            
            ### Normalization
            T_pulse = np.array(T_os_f) + np.array(T_so_f)
            T_pulse_samples = T_pulse*F_s/1000
            mT_pulse_samples = np.median(T_pulse_samples)
            AC_amp_med = np.median(A_AC)
            mAC_amp = np.median(AC_amp_med)

            AUCos_N = AUCos / (mT_pulse_samples * mAC_amp)
            AUCso_N = AUCso / (mT_pulse_samples * mAC_amp)
    return AUCos_N, AUCso_N, T_os_f, T_so_f, T_so_cd, A_AC, A_sp, A_off, mean_slope_os, mean_slope_so, PW_10, PW_25, PW_33, PW_50, PW_66, PW_75


#pulse_width_at_levelX: the width at X% of the SP Amplitude between the pulse onset and offset.
def PW_at_X_level(self, ppg_signal, On_sync_i,  SP_sync_i, Off_sync_i, On_sync_0, A_sp_i, F_s, X_percent): 
    ppg_signal = np.array(ppg_signal)
    X_percent_level = ppg_signal[On_sync_0] + (A_sp_i * X_percent/100)

    ix_X_os = np.argmin(np.abs(ppg_signal[On_sync_i:SP_sync_i]-X_percent_level))
    ix_X_so = np.argmin(np.abs(ppg_signal[SP_sync_i:Off_sync_i]-X_percent_level))
    
    # fig, ax = plt.subplots(1,3, figsize=(11,3))
    # ax[0].plot(ppg_signal[On_sync_i:SP_sync_i])
    # ax[0].plot(ix_X_os, ppg_signal[On_sync_i:SP_sync_i][ix_X_os],'bo')
    # ax[1].plot(ix_X_so, ppg_signal[SP_sync_i:Off_sync_i][ix_X_so],'ro')
    # ax[1].plot(ppg_signal[SP_sync_i:Off_sync_i])
    # ax[2].plot(ppg_signal[On_sync_i:Off_sync_i])
    # ax[2].plot(ix_X_os, ppg_signal[On_sync_i:Off_sync_i][ix_X_os],'bo')
    # ax[2].plot(SP_sync_i-On_sync_i+ix_X_so, ppg_signal[On_sync_i:Off_sync_i][SP_sync_i-On_sync_i+ix_X_so],'ro')

    PW_os_X = ((SP_sync_i - On_sync_i) - ix_X_os)*1000/F_s
    PW_so_X = (ix_X_so - 0)*1000/F_s
    PW_X = PW_os_X + PW_so_X
    return  PW_X, PW_os_X, PW_so_X

########################################## Respiratory features ###################################################


def calculate_inspiration_time(self, Resp_i, Resp_e, F_s):
    # Resp_i (troughs) are Resp_e (peaks) are onsets for inhalation and exhalation.
    Resp_i = Resp_i.to_numpy()
    Resp_e = Resp_e.to_numpy()
    
    inspiration_times = []
    if len(Resp_i) <= 1 or len(Resp_e) <= 1:
        return inspiration_times
    # Adjust Resp_i and Resp_e to start at the same cycle
    if Resp_i[0] > Resp_e[0]:
        Resp_e = Resp_e[1:]
    min_length = min(len(Resp_i), len(Resp_e))
    Resp_i = Resp_i[:min_length]
    Resp_e = Resp_e[:min_length]
    for i in range(min_length):
        insp_time = (Resp_e[i] - Resp_i[i]) / F_s
        inspiration_times.append(insp_time)
    return inspiration_times


def calculate_expiration_time(self, Resp_i, Resp_e, F_s): 
    # Resp_i (troughs) are Resp_e (peaks) are onsets for inhalation and exhalation.
    Resp_i = Resp_i.to_numpy()
    Resp_e = Resp_e.to_numpy()

    expiration_times = []
    
    if len(Resp_e) <= 1 or len(Resp_e) <= 1:
        return expiration_times
    if Resp_e[0] > Resp_i[0]:
        Resp_i = Resp_i[1:]
    min_length = min(len(Resp_i), len(Resp_e))  
    Resp_i = Resp_i[:min_length]
    Resp_e = Resp_e[:min_length]
    for i in range(min_length):
        exp_time = (Resp_i[i] - Resp_e[i]) / F_s
        expiration_times.append(exp_time)
    return expiration_times


def calculate_insp_exp_ratio(self, Resp_i, Resp_e, F_s):
    inspiration_time = self.calculate_inspiration_time(Resp_i, Resp_e, F_s)
    expiration_time = self.calculate_expiration_time(Resp_i, Resp_e, F_s)
    min_length = min(len(inspiration_time), len(expiration_time))
    insp_exp_rat = []
    for i in range(min_length):
        insp_time = inspiration_time[i]
        exp_time = expiration_time[i]
        if exp_time != 0:
            ie_ratio = insp_time / exp_time
            insp_exp_rat.append(ie_ratio)
        else:
            insp_exp_rat.append(np.nan)
    insp_exp_rat = np.array(insp_exp_rat)
    return insp_exp_rat




######################################### ABP features ###################################################
def ABP_features(self, abp_signal, SBP_samples, DBP_samples, F_s): #same cycle
    ABPonset = np.array(DBP_samples)
    onset = ABPonset[:-1]
    offset = ABPonset[1:]
    
    SBP_ix = np.array(SBP_samples)
    
    if len(SBP_ix) == 0:
        HR, aSBP, aDBP, MAP, PP, CO, TPR = np.nan*np.ones(7)
            
    else:
        ##step 1: find the correct pairings of onsets, SBP_ix, offsets in each abp pulse
        SBP_ix_sync = []
        On_sync = []
        Off_sync = []
        for k in range(len(onset)):
            cond1 = SBP_ix > onset[k]
            cond2 = SBP_ix < offset[k]
            SBP_ix_all = pd.Series(SBP_ix[np.where(cond1 & cond2)])
            if len(SBP_ix_all) != 0:
                SBP_ix_sync = np.append(SBP_ix_sync, SBP_ix_all[0])
                On_sync = np.append(On_sync, onset[k])
                Off_sync = np.append(Off_sync, offset[k])
        ##step 2: systole and diastole time durations (putting an HR-based constraint to avoid outliers)    
        SBP_ix_sync = np.array(SBP_ix_sync, dtype=int)
        On_sync = np.array(On_sync, dtype=int)
        Off_sync = np.array(Off_sync, dtype=int)


        CI = (Off_sync - On_sync)/F_s
        HR = 60/CI
    
        aSBP = abp_signal[SBP_ix_sync].values
        aDBP = abp_signal[On_sync].values
        
        MAP = (aSBP + 2*aDBP)/3 #mean arterial pressure
        PP = aSBP - aDBP #pulse pressure
        CO = PP * HR / 1000 #cardiac output (L/min.)
        TPR = MAP / CO #Total peripheral resistance (TPR) = MAP / CO
        ShockIndex = HR / aSBP
    return HR, aSBP, aDBP, MAP, PP, CO, TPR, ShockIndex






##FUNCTION: NN outlier correction
def NN_oulier_correction(self, signal, th=3):
    signal_samples1 = np.where(signal>(np.mean(signal)+th*np.std(signal)))[0]
    signal_samples2 = np.where(signal<(np.mean(signal)-th*np.std(signal)))[0]
    signal[signal_samples1] = np.nan
    signal[signal_samples2] = np.nan
    NN_corrected = pd.Series(signal).fillna(method='ffill')
    NN_corrected = NN_corrected.fillna(method='bfill')
    return NN_corrected.values



##FUNCTION: contributions of heart rate decelerations and accelerations to short- and long-term HRV 
def hra_acc_dec(self, NN):
    """ HRV- heart rate asymmetry analysis (HRA) - Asymmetry in Poincare plot
    Ref: Piskorski (2011), Asymmetric properties of long-term and total heart rate variability
    
    """
    N = len(NN) - 1
    x = NN[:-1]  # NN_n in x-axis
    y = NN[1:]  # NN_n+1 in y-axis

    D = y - x
    decelerate_ind = np.where(D > 0)[0]  # point-set above IL where y > x
    accelerate_ind = np.where(D < 0)[0]  # point-set below IL where y < x
    zerochange_ind = np.where(D == 0)[0]

    # Evaluation of distance to centroid line l2 
    D_l2_all = abs((x - np.mean(x)) + (y - np.mean(y))) / np.sqrt(2)

    # Distances to LI
    D_all = abs(y - x) / np.sqrt(2)

    # Short-term asymmetry (SD1)
    sd1d = np.sqrt(np.sum(D_all[decelerate_ind] ** 2) / (N - 1))
    sd1a = np.sqrt(np.sum(D_all[accelerate_ind] ** 2) / (N - 1))

    sd1I = np.sqrt(sd1d**2 + sd1a**2)
    C1_d = (sd1d / sd1I) ** 2
    C1_a = (sd1a / sd1I) ** 2
    SD1_d = sd1d  # SD1 deceleration
    SD1_a = sd1a  # SD1 acceleration
    
    
    # Long-term asymmetry (SD2)
    longterm_decelerate = np.sum(D_l2_all[decelerate_ind] ** 2) / (N - 1)
    longterm_accelerate = np.sum(D_l2_all[accelerate_ind] ** 2) / (N - 1)
    longterm_zerochange = np.sum(D_l2_all[zerochange_ind] ** 2) / (N - 1)

    sd2d = np.sqrt(longterm_decelerate + 0.5 * longterm_zerochange)
    sd2a = np.sqrt(longterm_accelerate + 0.5 * longterm_zerochange)

    sd2I = np.sqrt(sd2d**2 + sd2a**2)
    C2_d = (sd2d / sd2I) ** 2
    C2_a = (sd2a / sd2I) ** 2
    SD2_d = sd2d  # SD2 deceleration
    SD2_a = sd2a  # SD2 acceleration
    
    return C1_d, C1_a, SD1_d, SD1_a, C2_d, C2_a, SD2_d, SD2_a



#FUNCTION: HRV - DFA: Detrended fluctuation analysis of HRV
def hrv_dfa_calculation(self, NN, show_flag):
    """ HRV - DFA, Credit: Neurokit2. Detrended fluctuation analysis of HRV"""
    dfa_windows = [(4, 11), (12, None)]
    if dfa_windows[1][1] is None: max_beats = (len(NN) + 1) / 10  # Number of peaks divided by 10
    else: max_beats = dfa_windows[1][1]
    # No. of windows to compute for short and long term
    n_windows_short = int(dfa_windows[0][1] - dfa_windows[0][0] + 1)
    n_windows_long = int(max_beats - dfa_windows[1][0] + 1)
    # Compute DFA alpha1
    short_window = np.linspace(dfa_windows[0][0], dfa_windows[0][1], n_windows_short).astype(int) 
    DFA_alpha1, _ = nk.fractal_dfa(NN, multifractal=False, scale=short_window, show=show_flag) # For monofractal
    plt.show()
    # Compute DFA alpha2
    # sanatize max_beats
    if max_beats < dfa_windows[1][0] + 1:
        print("DFA_alpha2 related indices will not be calculated. "
            "The maximum duration of the windows provided for the long-term correlation is smaller "
            "than the minimum duration of windows. Refer to the `scale` argument in `nk.fractal_dfa()` "
            "for more information.")
        DFA_alpha2 = []
    else:
        long_window = np.linspace(dfa_windows[1][0], int(max_beats), n_windows_long).astype(int)
        DFA_alpha2, _ = nk.fractal_dfa(NN, multifractal=False, scale=long_window,  show=show_flag) # For monofractal
        plt.show()
    return DFA_alpha1, DFA_alpha2


def compute_gini_index(self, x):
    """
    Compute Gini index (0 to 1) for any real-valued array by shifting all values to be non-negative.
    Adds small epsilon to avoid division by zero.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]  # Remove NaNs and infs

        if len(x) < 2:
            return np.nan

        # Shift values if negatives are present
        if np.min(x) < 0:
            x = x - np.min(x)

        # Add small epsilon to ensure total sum is not zero
        x = x + 1e-7

        x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)

        gini = (2.0 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n
        return gini

    except Exception:
        return np.nan


    
def compute_entropy(self, x, bins=10):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    bins = min(bins, len(x))  # Prevent more bins than data points
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return scipy_entropy(hist)

def compute_slope(self, x):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return np.polyfit(np.arange(len(x)), x, 1)[0]  # Linear trend slope

def compute_autocorr(self, x, lag=1):
    x = x[~np.isnan(x)]
    if len(x) <= lag:
        return np.nan
    x_mean = np.mean(x)
    num = np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean))
    denom = np.sum((x - x_mean) ** 2)
    return num / denom if denom != 0 else np.nan

def compute_stat(self, val, stat_method):
    val = np.array(val)
    val = val[~np.isnan(val)]  # Remove NaNs

    if len(val) == 0:
        # Return a dict with appropriate structure depending on method
        if stat_method == 'all':
            return {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'mad': np.nan,
                'iqr': np.nan,
                'min': np.nan,
                'max': np.nan,
                'p25': np.nan,
                'p75': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'entropy': np.nan,
                'slope': np.nan,
                'autocorr_lag1': np.nan,
                'gini_index': np.nan,
                'valid_n': 0
            }
        else:
            return {stat_method: np.nan}
    
    if stat_method == 'mean':
        return {'mean':np.mean(val)}
    elif stat_method == 'median':
        return {'median':np.median(val)}
    elif stat_method == 'min':
        return {'min':np.min(val)}
    elif stat_method == 'max':
        return {'max':np.max(val)}
    elif stat_method == 'p25':
        return {'p25':np.percentile(val, 25)}
    elif stat_method == 'p75':
        return {'p75':np.percentile(val, 75)}
    elif stat_method == 'all':
        return {
            'mean': np.mean(val),
            'median': np.median(val),
            'std': np.std(val),
            'mad': np.mean(np.abs(val - np.median(val))),
            'iqr': iqr(val),
            'min': np.min(val),
            'max': np.max(val),
            'p25': np.percentile(val, 25),
            'p75': np.percentile(val, 75),
            'skewness': skew(val),
            'kurtosis': kurtosis(val),
            'entropy': self.compute_entropy(val),
            'slope': self.compute_slope(val),
            'autocorr_lag1': self.compute_autocorr(val),
            'gini_index': self.compute_gini_index(val),            
            'valid_n': len(val)
        }
    else:
        raise ValueError(f"Unknown stat_method: {stat_method}")

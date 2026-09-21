# -*- For plottig ECG and other physiological waveform data -*-
"""
Visualization of multimodal waveform data (e.g., ECG, PPG, Resp.) 

@author: Tilendra Choudhary
"""
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from plotly_resampler import register_plotly_resampler


###############################################
''' For Browser plotting if selecting Plotly. If you're 'using Jupyter NB, comment these lines '''
import plotly.io as pio
pio.renderers.default='browser'
###############################################


def MPT_plot_multiwav(
    signals: dict,
    peaks: dict = None,
    sampling_rate: float = None,
    timestamps = None,
    timestamp_resolution: str = None,
    method: str = "matplotlib",
    show_peaks: bool = True,
    figsize: tuple = (18.5, 10.5),
    width: float = 800,
    height: float = 440,
    sig_title: str = "All physiological waveforms with their fiducial points",
):
    """Generates plots for multimodal signals.

    Args:
        signals (dict): The dictionary of signals to be plotted.
        peaks (dict, optional): The dictionary of peaks to be plotted. Defaults to None.
        sampling_rate (float, optional): Sampling rate of the signal. Defaults to None.
        timestamps (ArrayLike, optional): Timestamp array. Defaults to None.
        timestamp_resolution (str, optional): Timestamp resolution. Defaults to None.
        method (str, optional): Package to generate plots. Defaults to 'matplotlib'.
        show_peaks (bool, optional): If True, peaks are plotted. Defaults to True.
        figsize (tuple, optional): Figure size for matplotlib. Defaults to (18.5, 10.5).
        width (float, optional): Figure width for Plotly. Defaults to 800.
        height (float, optional): Figure height for Plotly. Defaults to 440.

    Raises:
        ValueError: If timestamps is not None and timestamp resolution is not provided.
        ValueError: If timestamps array and the signal have different lengths.
        ValueError: If method is not 'matplotlib' or 'plotly'.
    """
    #ecg_sig = signals.get("ECG")
    sig_names = list(signals.keys())
    first_sig = signals.get(sig_names[0])

    if timestamps is not None:
        if len(timestamps) != len(first_sig):
            raise ValueError("Timestamps and the signal must have the same length!")

        if timestamp_resolution is None:
            raise ValueError("Timestamp resolution must be provided if timestamps are provided!")
        else:
            timestamp_resolution = timestamp_resolution

        x_values = timestamps
        x_label = "Time (" + timestamp_resolution + ")"

    else:
        if sampling_rate is not None:
            if timestamp_resolution is None:
                timestamp_resolution = "s"

            x_values = create_timestamp_signal(
                resolution=timestamp_resolution, length=len(first_sig), rate=sampling_rate, start=0
            )
            x_label = "Time (" + timestamp_resolution + ")"

        else:
            x_values = np.linspace(0, len(first_sig), len(first_sig))
            x_label = "Sample"

    if peaks is None:
        if show_peaks:
            raise ValueError("Peaks must be specified if show_peaks is True.")
        else:
            peaks = {}

    if method == "matplotlib":
        _plot_signal_matplotlib(
            signals=signals, peaks=peaks, x_values=x_values, x_label=x_label, figsize=figsize, show_peaks=show_peaks, plot_title=sig_title,
        )
    elif method == "plotly":
        _plot_signal_plotly(
            signals=signals,
            peaks=peaks,
            x_values=x_values,
            x_label=x_label,
            width=width,
            height=height,
            show_peaks=show_peaks,
            plot_title=sig_title,
        )
    else:
        raise ValueError("Undefined method.")



def _plot_signal_matplotlib(
    signals: dict,
    plot_title: str,
    peaks: dict = None,
    x_values: ArrayLike = None, 
    x_label: str = "Sample",
    figsize=(18.5, 10.5),
    show_peaks=True,
):
    """Generates plots for the signals using Matplotlib."""
    # Create figure
    fig, axs = plt.subplots(figsize=figsize)

    # Plot the signals and peaks
    for signal_name, signal in signals.items():

        if signal_name not in peaks.keys():
            peaks[signal_name] = {}

        create_signal_plot_matplotlib(
            ax=axs,
            signal=signal,
            x_values=x_values,
            show_peaks=show_peaks,
            peaks=peaks[signal_name],
            plot_title=" ",
            signal_name=signal_name,
            x_label=x_label,
        )

    fig.supxlabel(x_label)
    fig.supylabel("Amplitude")
    plt.title(plot_title)

    fig.tight_layout()
    plt.show()


def _plot_signal_plotly(
    signals: dict,
    plot_title: str,
    peaks: dict = None,
    x_values: ArrayLike = None, 
    x_label: str = "Sample",
    width=800,
    height=440,
    show_peaks=True,
):
    """Generates plots for signals using Plotly."""
    # Create figure
    if 'ABP' in list(signals.keys()):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    else:    
        fig = make_subplots(rows=1, cols=1)

    # Plot the signals and peaks
    for signal_name, signal in signals.items():

        if signal_name not in peaks.keys():
            peaks[signal_name] = {}
        
        # If ABP signal is present, plot it in the second row
        if signal_name != 'ABP':
            create_signal_plot_plotly(
                fig,
                signal=signal,
                x_values=x_values,
                show_peaks=show_peaks,
                peaks=peaks[signal_name],
                plot_title=" ",
                signal_name=signal_name,
                width=width,
                height=height,
                location=(1, 1),
            )
            fig.update_xaxes(title_text=x_label, row=1, col=1)
        else:
            create_signal_plot_plotly(
                fig,
                signal=signal,
                x_values=x_values,
                show_peaks=show_peaks,
                peaks=peaks[signal_name],
                plot_title=" ",
                signal_name=signal_name,
                width=width,
                height=height,
                location=(2, 1),
            )
            fig.update_xaxes(title_text=x_label, row=2, col=1)

    fig.update_layout(
        {"title": {"text": plot_title, "x": 0.45, "y": 0.9}}, xaxis_title=x_label, yaxis_title=" "
    )
    fig.show()

 
    
 
    
###########################################################################################    
########################## Utility Functions Related to Plot Tools ################################    
###########################################################################################

def create_signal_plot_plotly(
    fig: go.Figure,
    signal: ArrayLike = None,
    x_values: ArrayLike = None,
    show_peaks: bool = False,
    peaks: dict = None,
    plot_title: str = "Signal Plot",
    signal_name: str = "Signal",
    x_label: str = "Sample",
    width: float = 1050,
    height: float = 600,
    location: tuple = None,
):
    """Generates plots for given signals using Plotly.

    Args:
        fig (go.Figure): Figure to plot signal.
        signal (ArrayLike, optional): Array of y-axis values. Defaults to None.
        x_values (ArrayLike, optional): Array of x-axis values. Defaults to None.
        show_peaks (bool, optional): If True, peaks are plotted. Defaults to False.
        peaks (dict, optional): Dictionary of peaks to be plotted. Defaults to None.
        plot_title (str, optional): Plot title. Defaults to "Signal Plot".
        signal_name (str, optional): Name of signal to be plotted. Defaults to "Signal".
        x_label (str, optional): Label of x-axis. Defaults to 'Sample'.
        width (float, optional): Figure width. Defaults to 1050.
        height (float, optional): Figure height. Defaults to 600.
        location (tuple, optional): Subplot location. Defaults to None.

    Raises:
        ValueError: If location is not provided.
    """
    # adjust it
    limit = 200000

    if len(signal) > limit:
        Warning("Signal is too large and will be resampled. Consider using create_signal_plot instead")
        register_plotly_resampler(mode="auto")

    if x_values is None:
        x_values = np.linspace(0, len(signal), len(signal))

    if location is None:
        raise ValueError("Location must be specified")

    fig.append_trace(go.Scatter(x=x_values, y=signal, name=signal_name), row=location[0], col=location[1])

    if show_peaks:

        for peak_type, peak_loc in peaks.items():
            peak_amp = signal[peak_loc]
            fig.append_trace(
                go.Scatter(x=x_values[peak_loc], y=peak_amp, name=signal_name + " " + peak_type, mode="markers"),
                row=location[0],
                col=location[1],
            )

    fig.update_layout({"xaxis": {"range": [0, x_values.max()]}}, title=plot_title, width=width, height=height)
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=1.2,
        showarrow=False,
        text=plot_title,
        row=location[0],
        col=location[1],
    )

    
def create_signal_plot_matplotlib(
    ax: plt.Axes,
    signal: ArrayLike = None,
    x_values=None,
    show_peaks: bool = False,
    peaks: dict = None,
    plot_title: str = "Signal Plot",
    signal_name: str = "Signal",
    x_label: str = "Sample",
):
    """Generates plots for given signals using Matplotlib.

    Args:
        ax (plt.Axes): Axes to plot signal.
        signal (ArrayLike, optional): Array of y-axis values. Defaults to None.
        x_values (_type_, optional): Array of x-axis values. Defaults to None.
        show_peaks (bool, optional): If True, peaks are plotted. Defaults to False.
        peaks (dict, optional): Dictionary of peaks to be plotted. Defaults to None.
        plot_title (str, optional): Plot title. Defaults to "Signal Plot".
        signal_name (str, optional): Name of signal to be plotted. Defaults to "Signal".
        x_label (str, optional): Label of x-axis. Defaults to 'Sample'.
    """
    if x_values is None:
        x_values = np.linspace(0, len(signal), len(signal))

    # Check if there is existing legend
    if isinstance(ax.get_legend(), type(None)):
        legend = []
    else:
        legend = [x.get_text() for x in ax.get_legend().texts]

    ax.plot(x_values, signal)
    legend.append(signal_name)

    if show_peaks:
        for peak_type, peak_loc in peaks.items():
            peak_amp = signal[peak_loc]
            ax.scatter(x_values[peak_loc], peak_amp, label=peak_type)
            legend.append(signal_name + " " + peak_type)

    ax.set_title(plot_title)
    ax.set_xlim([0, max(x_values)])
    # ax.set_xlabel(x_label)
    # ax.set_ylabel('Amplitude')

    ax.legend(legend, loc="center left", bbox_to_anchor=(1.0, 0.5))
    


    
    
###########################################################################################    
########################## Utility Functions Related to Time Tools ################################    
###########################################################################################

def create_timestamp_signal(resolution: str, length: float, start: float, rate: float) -> ArrayLike:
    """Generates a timestamp array.

    Args:
        resolution (str): Timestamp resolution. It can be 'ns', 'ms', 's' or 'min'.
        length (float): Length of timestamp array to be generated.
        start (float): Starting time.
        rate (float): Rate of increment.

    Raises:
        ValueError: If starting time is less then zero.
        ValueError: If resolution is undefined.

    Returns:
        ArrayLike: Timestamp array.
    """

    if start < 0:
        raise ValueError("Timestamp start must be greater than 0")

    if resolution == "ns":
        timestamp_factor = 1 / 1e-9
    elif resolution == "ms":
        timestamp_factor = 1 / 0.001
    elif resolution == "s":
        timestamp_factor = 1
    elif resolution == "min":
        timestamp_factor = 60
    else:
        raise ValueError('resolution must be "ns","ms","s","min"')

    timestamp = (np.arange(length) / rate) * timestamp_factor
    timestamp = timestamp + start

    return timestamp



def check_timestamp(timestamp, timestamp_resolution):

    possible_timestamp_resolution = ["ns", "ms", "s", "min"]

    if timestamp_resolution in possible_timestamp_resolution:
        pass
    else:
        raise ValueError('timestamp_resolution must be "ns","ms","s","min"')

    if np.any(np.diff(timestamp) < 0):
        raise ValueError("Timestamp must be monotonic")

    return True











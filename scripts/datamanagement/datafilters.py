import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt


def radius_filter(data: np.ndarray, radius: int) -> np.ndarray:
    """
    compute mean of data in an interval [i-radius, i+radius]

    :param data: numpy array
    :param radius: interval to compute a mean
    :return: filtered numpy array
    """
    filtered = np.copy(data)
    for i in range(radius, len(data) - radius - 1):
        filtered[i] = data[i - radius: i + radius + 1].sum(axis=0) / (2*radius + 1)
    return filtered


def butter_lowpass_filtfilt(data: np.ndarray, smoothness: float = 0.02, order: int = 5) -> np.ndarray:
    """
    smooth data

    :param data: numpy array
    :param smoothness: the lower this number is the smoother result you get
    :param order: the order of the filter
    :return: filtered data. numpy array
    """
    b, a = butter(order, smoothness, btype='low', analog=False)
    return filtfilt(b, a, data)


def smooth_data(data: np.ndarray, smoothness: float = 0.02):
    """
    smooth all dimensions with data of same type, e.g. x, y dimensions for velocity

    :param data: numpy array of a shape (r, c)
    :param smoothness: the lower this number is the smoother result you get
    :return: smoothed data
    """
    for i in range(data.shape[1]):
        data[:, i] = butter_lowpass_filtfilt(data[:, i], smoothness=smoothness)
    return data


def applyfilter(params: dict, data: np.ndarray) -> np.ndarray:
    """
    filter data if stated according to a config

    :param params: parameters of data preprocessing
    :param data: numpy array
    :return: filtered numpy array
    """
    if params['filter'] == "mean":
        data = radius_filter(data=data, radius=params['filter_radius'])
    elif params['filter'] == "wiener":
        data = signal.wiener(data, params['wiener_window'])
    elif params['filter'] == "meanwiener":
        data = radius_filter(data=data, radius=params['filter_radius'])
        data = signal.wiener(data, params['wiener_window'])
    elif params['filter'] == "smooth":
        data = smooth_data(data=data, smoothness=params['smoothness'])
    return data

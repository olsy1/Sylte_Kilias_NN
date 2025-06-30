import numpy as np
from sklearn import preprocessing
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import get_data


def get_moving_frames(speed, threshold=0.01, exclude_frames_after=None):

    moving = (speed >= threshold) | np.isnan(speed)
    
    if exclude_frames_after is not None and exclude_frames_after > 0:

        low_speed = speed < threshold
        high_speed = speed >= threshold
        
        valid_transitions = np.where(
            (np.diff(high_speed.astype(int)) == 1) &  
            (~np.isnan(speed[:-1])) &                
            (~np.isnan(speed[1:]))                    
        )[0] + 1
        

        for trans_idx in valid_transitions:
            end_idx = min(trans_idx + exclude_frames_after, len(moving))
            moving[trans_idx:end_idx] = False
    
    return moving

    

def calculate_speed(position, time_interval=1/15, sigma=2, threshold=1.98):
    speed = np.full_like(position, np.nan, dtype=float)
    run_starts, run_ends = get_data.identify_runs2(position, threshold)
    for start, end in zip(run_starts, run_ends):
        trial_pos = position[start:end+1]
        
        delta_pos = np.diff(trial_pos) 
        trial_speed = delta_pos / time_interval
        trial_speed = np.insert(trial_speed, 0, np.nan)
        trial_speed_smooth = gaussian_filter1d(trial_speed, sigma=sigma)
        speed[start:end+1] = trial_speed_smooth
    
    speed[speed<0] =0

    return speed
def filter_calcium_trace(trace, min_events):
    filtered_trace = trace.copy()

    for i in range(trace.shape[0]):
        transitions = np.diff((trace[i] > 0).astype(int))
        num_events = np.sum(transitions == 1)

        if num_events < min_events:
            filtered_trace[i, :] = 0
    
    return filtered_trace


def filter_calcium_trace2(trace, min_events):
    filtered_trace = trace.copy()
    for i in range(trace.shape[0]):
        diff = np.diff(trace[i])
        num_events = np.sum(diff > 0)
        if num_events < min_events:
            filtered_trace[i, :] = 0
   
    return filtered_trace


def filter_calcium_trace_percentage(trace, min_percentage):
    filtered_trace = trace.copy()
    for i in range(trace.shape[0]):
        positive_frames = np.sum(trace[i] > 0)
        total_frames = trace.shape[1]
        percentage_positive = (positive_frames / total_frames) * 100
        if percentage_positive < min_percentage:
            filtered_trace[i, :] = 0

    return filtered_trace


def soft_normalize(response, normalization_factor=5):
    return response / (np.ptp(response) + normalization_factor)

def normalize_calcium_data(data, normalization_factor=20):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        normalized_data[i] = soft_normalize(data[i], normalization_factor)
    return normalized_data

def robust_normalize(data):
    median = np.median(data, axis=1, keepdims=True)
    iqr = np.percentile(data, 75, axis=1, keepdims=True) - np.percentile(data, 25, axis=1, keepdims=True)
    return (data - median) / (iqr + 1e-6)

def standardize_transients(transients, method='standard'):
    if method == 'standard':
        scaler = preprocessing.StandardScaler(with_mean=False).fit(transients)
    elif method == 'standard_m':
        scaler = preprocessing.StandardScaler(with_mean=True).fit(transients)
    elif method == 'quantile':
        scaler = preprocessing.QuantileTransformer(n_quantiles=100).fit(transients)
    elif method == 'minmax':
        scaler = preprocessing.MinMaxScaler().fit(transients)
    elif method == 'robust':
        scaler = preprocessing.RobustScaler().fit(transients)
    elif method == 'normalize':
        scaler = preprocessing.Normalizer().fit(transients)
    elif method == 'powerTransformer':
        scaler = preprocessing.PowerTransformer().fit(transients)
    elif method == 'soft_normalize':
        return normalize_calcium_data(transients, normalization_factor=5)
    elif method=='robust_norm':
        return robust_normalize(transients)
    else:
        raise ValueError("Invalid standardization method.")
   
    return scaler.transform(transients)



def convolve(transients, width):
    n = 0
    for neuron in transients:
        spike_train_1_neuron = transients[n,:]
        kernel = signal.windows.gaussian(len(spike_train_1_neuron), width)
        transients[n,:] = signal.fftconvolve(spike_train_1_neuron, kernel, mode='same')
        n+=1
    return transients 


def convolve1d(trace, width):
    kernel = signal.windows.gaussian(len(trace), width)
    trace = signal.fftconvolve(trace, kernel, mode='same')

    return trace 

def digitize(data, bins=40):
    return np.digitize(data,np.linspace(0,1,bins) )

def mean_bins(transients, digitized):
    bins = np.max(digitized)
    e=0
    transients_mean = np.zeros((len(transients), bins))
    for i in range(1,bins+1):
        transients_mean[:,e] = np.mean(transients[:, digitized == i],1)
        e+=1
    return transients_mean

def scale_and_digitize(data, min_val=1, max_val=10, num_bins=10):
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * (max_val - min_val) + min_val
    bins = np.linspace(min_val, max_val, num_bins + 1)
    digitized_data = np.digitize(scaled_data, bins, right=True)
    return scaled_data, digitized_data




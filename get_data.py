import copy
import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy import spatial
import numpy as np
from scipy import spatial
import copy
import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy import spatial
import population_activity as pop
import helper_functions as hf
import transients




def trial_one_session_AK(datapath, session, context):
    dp = AKDataProcessor(datapath, session, context)
    dp.open_h5py()
    hist, hist_norm =average_spatial_tuning_function(dp.calcium, dp.pos)
    hist_sort_map = sort_map(hist_norm)
    return {'calcium': dp.calcium, 'pos': dp.pos, 'hist': hist, 'hist_norm': hist_norm, 'hist_sort_map': hist_sort_map}


class AKDataProcessor:
    def __init__(self, datapath, session, context):
        self.datapath = datapath
        self.session = session
        self.data = None
        self.context = context
    
    def open_h5py(self):
        self.calcium = []
        self.pos = []
        with h5py.File(self.datapath, 'r') as f:
            d = f[str(self.session) + '/' + self.context]
            runs = list(d.keys())
            for run in runs:
                self.calcium.append(np.array(d[run][:]))
                for _, value in d[run].attrs.items():
                    self.pos.append(value)
            self.calcium = np.vstack(self.calcium).T
            self.pos = np.hstack(self.pos)


            basic_ind = (self.pos > 0.099) & (self.pos < 1.98)

            #basic_ind = (self.pos > 0.0227) & (self.pos < 1.919)

            drops = np.where(np.diff(self.pos) < -0.5)[0] + 1  
            exclude_indices = []
            for drop in drops:
                exclude_indices.extend(range(drop, min(drop + 4, len(self.pos))))  


            ind = basic_ind & ~np.isin(np.arange(len(self.pos)), exclude_indices)

            self.calcium = self.calcium[:, ind]
            self.pos = self.pos[ind]


    def get_transients(self):
        mask = transients.transient_mask(self.calcium)
        transient_rise = transients.transient_rise(self.calcium, mask)
        self.calcium = transients.calculate_transient_rate(transient_rise, window_size=10)



def rescale_pos(data, new_min, new_max):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

def circular_place_map(transients, lin_pos, bins=20):
    X = copy.deepcopy(transients)

    norm_pos = (lin_pos - np.min(lin_pos)) / (np.max(lin_pos) - np.min(lin_pos))
    angular_pos = norm_pos * 2 * np.pi
    bin_edges = np.linspace(0, 2 * np.pi, bins + 1)
    
    digitized_pos = np.digitize(angular_pos, bin_edges) - 1
    
    n_neurons = transients.shape[0]
    circular_hist = np.zeros((n_neurons, bins))
    circular_hist_norm = np.zeros((n_neurons, bins))
    occupancy, _ = np.histogram(angular_pos, bins=bin_edges)
    occupancy = occupancy / np.sum(occupancy)
    

    for i in range(n_neurons):
        for j in range(bins):
            mask = digitized_pos == j
            if np.sum(mask) > 0:
                circular_hist[i, j] = np.mean(transients[i, mask])
        

        circular_hist[i, :] /= occupancy
        min_val = np.min(circular_hist[i, :])
        max_val = np.max(circular_hist[i, :])
        circular_hist_norm[i, :] = (circular_hist[i, :] - min_val) / (max_val - min_val)
    
    return circular_hist, angular_pos, digitized_pos





def average_spatial_tuning_function(transients, lin_pos, bins=20, z_score=False):
    X=copy.deepcopy(transients)#np.reshape(transients,[len(transients),-1])
    #if z_score ==True:
    #X=stats.zscore(X,axis=1)
    hist, _, _ = stats.binned_statistic(lin_pos, X, statistic= 'mean', bins=bins)
    hist = np.nan_to_num(hist, nan=0.0)

    #occ,a=np.histogram(lin_pos,bins=bins) # Get occupancy map
    #occ=occ/np.sum(occ) # Convert to probability.
    #hist=hist/occ
    hist_norm=np.empty_like(hist)
    for n in range(len(hist)):
        hist_norm[n]= rescale_pos(hist[n],0,1)
    return hist, hist_norm

def average_spatial_tuning_function3(transients, lin_pos, bins=20):
    X=copy.deepcopy(transients)#np.reshape(transients,[len(transients),-1])
    X=stats.zscore(X,axis=1)
    hist, _, _ = stats.binned_statistic(lin_pos, X, statistic='mean', bins=bins)
    occ,a=np.histogram(lin_pos,bins=bins) # Get occupancy map
    occ=occ/np.sum(occ) # Convert to probability.
    hist=hist/occ
    hist_norm=np.empty_like(hist)
    for n in range(len(hist)):
        hist_norm[n]= rescale_pos(hist[n],0,1)
    return hist_norm, 0

def average_spatial_tuning_function2(transients, lin_pos, bins=20):
    X=copy.deepcopy(transients)#np.reshape(transients,[len(transients),-1])
    hist, _, _ = stats.binned_statistic(lin_pos, X, statistic='mean', bins=bins)
    return hist


def rescale_pos(values, new_min = 0, new_max= 150):
    output = []
    old_min, old_max = min(values), max(values)
    with np.errstate(divide='ignore', invalid='ignore'):
        for v in values:
            new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
            output.append(new_v)

    return output


def sort_map(hist):
    argmax_values = np.argmax(hist, axis=1)  
    max_values = np.max(hist, axis=1)        
    zero_neurons = max_values == 0
    argmax_values[zero_neurons] = -1
    sort_map = np.argsort(argmax_values)    
    return sort_map

def identify_runs3(lin_pos, threshold=1.9):
    crossings = np.where(np.diff(np.sign(lin_pos - threshold)) == -2)[0] + 1
    run_starts = np.insert(crossings, 0, 0) if crossings[0] != 0 else crossings
    run_ends = np.append(run_starts[1:]-1, len(lin_pos)-1)
    
    return run_starts, run_ends


def identify_runs2(lin_pos, threshold=1.9):
    run_starts = np.where(np.diff(np.sign(lin_pos - threshold)) == -2)[0] + 1
    if lin_pos[0] < threshold:
        run_starts = np.insert(run_starts, 0, 0)
    run_ends = np.append(run_starts[1:]-1, len(lin_pos)-1)  
    
    return run_starts, run_ends 

def identify_runs(lin_pos, threshold=1.9):
    run_starts = np.where(np.diff(np.sign(lin_pos - threshold)) == -2)[0] + 1
    if lin_pos[0] < threshold:
        run_starts = np.insert(run_starts, 0, 0)
    run_ends = np.append(run_starts[1:]-1, len(lin_pos))
    return run_starts[:-1], run_ends[:-1]

def separate_odd_even_runs(transients, lin_pos, threshold=1.9):
    run_starts, run_ends = identify_runs(lin_pos, threshold)

    transients_odd = []
    lin_pos_odd = []
    transients_even = []
    lin_pos_even = []

    for i in range(len(run_starts)):
        if i % 2 == 0:
            transients_odd.append(transients[:,run_starts[i]:run_ends[i]])
            lin_pos_odd.append(lin_pos[run_starts[i]:run_ends[i]])
        else:
            transients_even.append(transients[:,run_starts[i]:run_ends[i]])
            lin_pos_even.append(lin_pos[run_starts[i]:run_ends[i]])
            
    return np.hstack(transients_odd), np.hstack(lin_pos_odd), np.hstack(transients_even), np.hstack(lin_pos_even)


def split_half_data(transients, lin_pos):
    half_len = int(np.round(len(transients[0])/2))
    transients1 = transients[:,:half_len]
    transients2 = transients[:,half_len:]
    lin_pos1 = lin_pos[:half_len]
    lin_pos2 = lin_pos[half_len:]
    return transients1, lin_pos1, transients2, lin_pos2

def bin_runs(transient_rates, lin_pos, num_bins=20):
    run_starts, run_ends = identify_runs(lin_pos)
    num_neurons = transient_rates.shape[0]
    binned_data = []

    for start, end in zip(run_starts, run_ends):
        run_pos = lin_pos[start:end]
        run_transients = transient_rates[:, start:end]
        
        bins = np.linspace(np.min(run_pos), np.max(run_pos), num_bins + 1)
        digitized = np.digitize(run_pos, bins) - 1 
        digitized[digitized == num_bins] = num_bins - 1  

        #calculate mean transient rate for each bin
        binned_transients = np.zeros((num_neurons, num_bins))
        for bin_idx in range(num_bins):
            bin_mask = digitized == bin_idx
            if np.any(bin_mask):
                binned_transients[:, bin_idx] = np.mean(run_transients[:, bin_mask], axis=1)
        
        binned_data.append(binned_transients)


    binned_data = np.stack(binned_data, axis=0)  #shape (num_runs, num_neurons, num_bins)
    return binned_data




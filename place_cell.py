import numpy as np
from scipy import stats
import decoding2
import copy
import random



np.random.seed(42)
random.seed(42)


def average_spatial_tuning_function(transients, lin_pos, bins=20):
    X = copy.deepcopy(transients)
    hist = np.zeros((X.shape[0], bins))
    bin_edges = np.linspace(np.min(lin_pos), np.max(lin_pos), bins + 1)
    
    for i in range(X.shape[0]):
        hist[i], _ = np.histogram(lin_pos, bins=bin_edges, weights=X[i], density=False)
        occupancy, _ = np.histogram(lin_pos, bins=bin_edges)
        valid_bins = occupancy > 0
        hist[i, valid_bins] = hist[i, valid_bins] / occupancy[valid_bins]

    
    return hist



def spatial_information_rate_map(rate_map, occupancy, fps=15):
    occupancy = np.array(occupancy, dtype=float)
    occupancy = occupancy / np.sum(occupancy)
    
    mean_rate = np.sum(rate_map * occupancy)
    
    if mean_rate <= 0:
        return 0.0
    
    valid_bins = (occupancy > 0) & (rate_map > 0)
    if not valid_bins.any():
        return 0.0
    
    # Calculate Skaggs information
    rate_ratio = rate_map[valid_bins] / mean_rate
    information = np.sum(
        occupancy[valid_bins] * rate_map[valid_bins] * np.log2(rate_ratio)
    )
    
    return information * fps

def identify_place_cells(transients, lin_pos, bins=20, shuffles=100, 
                        threshold=99, min_transients=1, fps=15, shuffle_seed=42):

    rng = np.random.default_rng(shuffle_seed)
    shuffle_seeds = rng.integers(0, 2**32, size=shuffles)
    
    hist_occ = average_spatial_tuning_function(transients, lin_pos, bins)
    occ, _ = np.histogram(lin_pos, bins=bins)
    occ = occ / np.sum(occ)
    
    spatial_info_scores = np.zeros(transients.shape[0])
    valid_cells = []
    
    for i in range(transients.shape[0]):
        if np.sum(transients[i] > 0) >= min_transients:
            rate_map = hist_occ[i]
            spatial_info = spatial_information_rate_map(rate_map, occ, fps)
            spatial_info_scores[i] = spatial_info
            valid_cells.append(i)
    shuffled_info_scores = []
    for shuffle_idx in range(shuffles):
        shuffled_transients = decoding2.circular_shuffle_features(
            transients.T, 
            lin_pos, 
            seed=shuffle_seeds[shuffle_idx]
        ).T  
        
        shuffled_hist_occ = average_spatial_tuning_function(
            shuffled_transients, lin_pos, bins
        )
        

        for i in valid_cells:
            shuffled_rate_map = shuffled_hist_occ[i]
            shuffled_info = spatial_information_rate_map(
                shuffled_rate_map, occ, fps
            )
            shuffled_info_scores.append(shuffled_info)
    

    threshold_value = np.percentile(shuffled_info_scores, threshold)
    

    place_cells = [i for i in valid_cells 
                  if spatial_info_scores[i] > threshold_value]
    
    return place_cells, spatial_info_scores#, threshold_value

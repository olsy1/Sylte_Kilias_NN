import numpy as np
import get_data
import matplotlib.pyplot as plt
import population_activity as pop
import helper_functions as hf
import copy
import place_cell as pc
from scipy.linalg import norm
import simulate_drift as sim
#import sub_manifold as sm
import decoding2 as decoding2
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import normalize
from scipy import stats
import random

np.random.seed(1)
random.seed(1)
def trial_one_session_AK(datapath, session, Context, remove_inactive=False, transients=False, 
                         dim_red=False, shuffle=False, standardize='stand', conv=False, remove_day_inactive=False, 
                         remove_stable=False, remove_stable_tresh=0.7, indices=False):
    dp = get_data.AKDataProcessor(datapath, session, Context)
    dp.open_h5py()
    dp.calcium = dp.calcium[:,30:]
    dp.pos = dp.pos[30:]
    
    if conv == True:
        dp.calcium = hf.convolve(dp.calcium,10)

    speed = hf.calculate_speed(dp.pos)
    moving = hf.get_moving_frames(speed, threshold=0.025, exclude_frames_after=None)
    dp.calcium = dp.calcium[:, moving]
    dp.pos = dp.pos[moving]
    

    dp.calcium = hf.filter_calcium_trace_percentage(dp.calcium,2)

    if remove_inactive is not False and remove_inactive.size > 0:
       dp.calcium = dp.calcium[remove_inactive]
    if remove_day_inactive is not False:
        dp.calcium = dp.calcium[dp.calcium.sum(axis=1) > 0]

    if shuffle == True:
        dp.calcium = decoding2.circular_shuffle_features(dp.calcium.T).T

    
    
    if transients == True:
        dp.get_transients()

    if indices is not False:
        dp.calcium = dp.calcium[indices]

    if remove_stable is not False:
        dp.calcium = dp.calcium[(remove_stable <= remove_stable_tresh) | np.isnan(remove_stable)]

    if standardize == 'std':
        dp.calcium=normalize.normalize_by_std(dp.calcium)
    elif standardize == 'stand':
        dp.calcium = hf.standardize_transients(dp.calcium.T, method='standard').T
    elif standardize == 'stand_m':
        dp.calcium = hf.standardize_transients(dp.calcium.T, method='standard_m').T
    elif standardize == 'stand_t':
        dp.calcium = hf.standardize_transients(dp.calcium, method='standard_m')
    elif standardize == 'quantile':
        dp.calcium = hf.standardize_transients(dp.calcium.T, method='quantile').T
    elif standardize == 'quantile_t':
        dp.calcium = hf.standardize_transients(dp.calcium, method='quantile')
    elif standardize == 'quantile_stand':
        dp.calcium = hf.standardize_transients(dp.calcium.T, method='quantile').T
        dp.calcium = hf.standardize_transients(dp.calcium.T, method='standard_m').T
    elif standardize == 'minmax':
        dp.calcium = hf.standardize_transients(dp.calcium.T, method='minmax').T
    elif standardize == 'soft_normalize':
        dp.calcium = hf.standardize_transients(dp.calcium.T, method='soft_normalize').T
    elif standardize == 'robust_norm':
        dp.calcium = hf.standardize_transients(dp.calcium, method='robust_norm')
    
    dp.calcium = np.nan_to_num(dp.calcium, nan=0.0)



    if dim_red == 'dpca':
        dim_red = pop.TransientEmbedding(dp.calcium.T, method='dpca', n_components=10)
        dp.calcium  = dim_red.embedding(pos= dp.pos)

    if dim_red == 'pca':
        dim_red = pop.TransientEmbedding(dp.calcium.T, method='pca', n_components=10)
        dp.calcium = dim_red.embedding().T

    hist, hist_norm =get_data.average_spatial_tuning_function(dp.calcium, dp.pos)
    hist_sort_map = get_data.sort_map(hist)

    return {'calcium': dp.calcium, 'pos': dp.pos, 'hist': hist, 'hist_norm': hist_norm, 'hist_sort_map': hist_sort_map}




def get_active_neurons_per_day(datapath, sessions, Context, standardize='stand'):
    active = [None] * len(sessions)
    for i, session in enumerate(sessions):
        trial_data = trial_one_session_AK(datapath, session, Context, transients=False, dim_red=False, shuffle=False, standardize='None')
        active[i] = np.max(trial_data['calcium'], axis=1) > 0
    active = np.vstack(active).T
    active_all = np.sum(active, axis=1) >0# == len(active[0])

    return active, active_all


def decode_position(datapath, reference, sessions, Context, model='linear', use_circular=False, align_sessions=False, transients=False, dim_red=False, shuffle=False, standardize='stand', remove_inactive=False):
    if remove_inactive is not False:
        _,remove_inactive = get_active_neurons_per_day(datapath, [reference] + sessions, Context, standardize=standardize)
    ref = trial_one_session_AK(datapath, reference, Context, dim_red=dim_red, transients=transients, shuffle=False, remove_inactive=remove_inactive, standardize=standardize)
    
    data_list = []
    
    X = ref['calcium'].T
    y = ref['pos']

    for session in sessions:
        if align_sessions:
            _, _, _, _, rotated, pos, X = AK_align_2_sets2(datapath, reference, session, Context, dim_red, transients, shuffle=False, standardize=standardize)
            data_list.append({'calcium': rotated, 'pos': pos})

        else:
            data = trial_one_session_AK(datapath, session, Context, dim_red=dim_red, transients=transients, shuffle=False, remove_inactive=remove_inactive, standardize=standardize)
            data_list.append(data)
    

    X_new_list = [data['calcium'].T for data in data_list]
    y_new_list = [data['pos'] for data in data_list]
    
    return decoding2.decode_neural_activity(X, y, X_new_list, y_new_list, model, use_shuffled=shuffle, shuffle_seed=0)



def decode_position_between(datapath, reference, reference_context, session, Context, model='linear', transients=False, dim_red=False, shuffle=False, standardize='stand'):
    ref = trial_one_session_AK(datapath, reference, reference_context, dim_red=dim_red, transients=transients, shuffle=False, standardize=standardize)
    
    X = ref['calcium'].T
    y = ref['pos']

    data = trial_one_session_AK(datapath, session, Context, dim_red=dim_red, transients=transients, shuffle=False, remove_inactive=False, standardize=standardize)

    X_new_list = [data['calcium'].T]
    y_new_list = [data['pos']]
    
    return decoding2.decode_neural_activity(X, y, X_new_list, y_new_list, model, use_shuffled=shuffle, shuffle_seed=0)


def sort_maps_from_reference_AK(datapath, Context, reference, maps, reference_type='no_reference', hist='hist', transients=False, shuffle=False,
                                 dim_red=False, standardize='stand', remove_inactive=False, remove_day_inactive=False, remove_stable=False, remove_stable_tresh=0.7):        
    
    if remove_inactive is not False:
        _,remove_inactive = get_active_neurons_per_day(datapath, [reference] + maps, Context, standardize=standardize)

    ref = trial_one_session_AK(datapath, reference, Context, dim_red=dim_red, transients=transients, shuffle=shuffle, 
                               remove_inactive=remove_inactive, standardize=standardize, remove_day_inactive=remove_day_inactive,
                                 remove_stable=remove_stable, remove_stable_tresh=remove_stable_tresh)
    hist_sorts = [None] * (len(maps)+1)

    if reference_type == 'no_reference':
        hist_sorts[0] = ref[hist]
    else:
        hist_sorts[0] = ref[hist][ref['hist_sort_map']]

    for i, map in enumerate(maps):
        i+=1
        data = trial_one_session_AK(datapath, map, Context, dim_red=dim_red, transients=transients, shuffle=shuffle, 
                                    remove_inactive=remove_inactive, standardize=standardize, remove_day_inactive=remove_day_inactive,
                                    remove_stable=remove_stable, remove_stable_tresh=remove_stable_tresh)
        if reference_type == 'no_reference':
            hist_sorts[i] = data[hist]
        if reference_type == 'reference':
            hist_sorts[i] = data[hist][ref['hist_sort_map']]
        elif reference_type == 'own_reference':
            hist_sorts[i] = data[hist][data['hist_sort_map']]
    return hist_sorts



def get_components(datapath, Context, reference, maps, dim_red='dpca', transients=False, shuffle=False, standardize='stand'):        
    ref = trial_one_session_AK(datapath, reference, Context,transients=transients, shuffle=shuffle, standardize=standardize)
    components = [None] * (len(maps)+1)

    translation_scale = pop.ManifoldAlignment.get_scaling_translation(ref['calcium'].T)
    ref['calcium'] = pop.ManifoldAlignment.apply_scaling_translation(ref['calcium'].T, translation_scale[0], translation_scale[1]).T

    emb = pop.TransientEmbedding(ref['calcium'].T, method=dim_red, n_components=10)
    _ = emb.embedding(ref['pos'])
    components[0] = emb.components()

    for i, map in enumerate(maps):
        i+=1
        data = trial_one_session_AK(datapath, map, Context, transients=transients, shuffle=shuffle)
        translation_scale = pop.ManifoldAlignment.get_scaling_translation(data['calcium'].T)
        data['calcium'] = pop.ManifoldAlignment.apply_scaling_translation(data['calcium'].T, translation_scale[0], translation_scale[1]).T
        emb = pop.TransientEmbedding(data['calcium'].T, method=dim_red, n_components=10)
        _ = emb.embedding(data['pos'])
        components[i] = emb.components()

    return components



def sort_maps_from_reference_AK_shuffle(datapath, Context, reference, maps, reference_type='reference', hist='hist',
                                         transients=False, shuffle=False, dim_red=False, standardize='stand', shuffle_type='perm', remove_inactive=False):        
    
    if remove_inactive is not False:
        _,remove_inactive = get_active_neurons_per_day(datapath, [reference] + maps, Context, standardize=standardize)

    ref = trial_one_session_AK(datapath, reference, Context, dim_red=dim_red, transients=transients, shuffle=shuffle, remove_inactive=remove_inactive, standardize=standardize)
    if shuffle_type == 'perm':
        ref['hist'] = np.random.permutation(ref['hist'].T)
    else:
        shuff_cal = decoding2.circular_shuffle_features(ref['calcium'].T, ref['pos']).T
        ref['hist'],_ = get_data.average_spatial_tuning_function(shuff_cal, ref['pos'])

    hist_sorts = [None] * (len(maps)+1)

    if reference_type == 'no_reference':
        hist_sorts[0] = ref[hist]
    else:
        hist_sorts[0] = ref[hist][ref['hist_sort_map']]

    for i, map in enumerate(maps):
        i+=1
        data = trial_one_session_AK(datapath, map, Context, dim_red=dim_red, transients=transients, shuffle=shuffle, remove_inactive=remove_inactive,   standardize=standardize)
        if shuffle_type == 'perm':
            data['hist'] = np.random.permutation(data['hist'].T).T
        else:
            shuff_cal = decoding2.circular_shuffle_features(data['calcium'].T, data['pos']).T
            data['hist'],_ = get_data.average_spatial_tuning_function(shuff_cal, data['pos'])

        if reference_type == 'no_reference':
            hist_sorts[i] = data[hist]
        if reference_type == 'reference':
            hist_sorts[i] = data[hist][ref['hist_sort_map']]
        elif reference_type == 'own_reference':
            hist_sorts[i] = data[hist][data['hist_sort_map']]
    return hist_sorts



def circle_shuffle_two_sessions_hist(datapath, session1, session2, Context,  dim_red=False, remove_inactive=False, standardize='stand'):
    if remove_inactive is not False:
        _,remove_inactive = get_active_neurons_per_day(datapath, remove_inactive, Context, standardize=standardize)
    data1 = trial_one_session_AK(datapath, session1, Context, shuffle=False, dim_red=dim_red, remove_inactive=remove_inactive, standardize=standardize)
    data2 = trial_one_session_AK(datapath, session2, Context,shuffle=False, dim_red=dim_red, remove_inactive=remove_inactive, standardize=standardize)    
    hist2 = decoding2.circular_shuffle_features(data2['hist'].T).T

    return data1['hist'], hist2

def shuffle_two_sessions_hist(datapath, session1, session2, Context,  dim_red=False, remove_inactive=False, standardize='stand', remove_day_inactive=False):
    if remove_inactive is not False:
        _,remove_inactive = get_active_neurons_per_day(datapath, remove_inactive, Context, standardize=standardize)
    data1 = trial_one_session_AK(datapath, session1, Context, shuffle=False, dim_red=dim_red, remove_inactive=remove_inactive, standardize=standardize, remove_day_inactive=remove_day_inactive)
    data2 = trial_one_session_AK(datapath, session2, Context,shuffle=False, dim_red=dim_red, remove_inactive=remove_inactive, standardize=standardize, remove_day_inactive=remove_day_inactive)
    
    #data1['hist'] = decoding2.circular_shuffle_features(data1['hist'].T).T
    #data2['hist'] = decoding2.circular_shuffle_features(data2['hist'].T).T
    #data1['hist'] = decoding2.circular_shuffle_features(data1['hist'])
    #data2['hist'] = decoding2.circular_shuffle_features(data2['hist'])

    hist1 = np.random.permutation(data1['hist'])
    hist2 = np.random.permutation(data2['hist'])
    hist1=np.random.permutation(hist1.T).T
    hist2= np.random.permutation(hist2.T).T

    return hist1, hist2


def shuffle_two_sessions_hist_sim(hist1x, hist2x):
    hist1 = np.random.permutation(hist1x)
    hist2 = np.random.permutation(hist2x)
    hist1= np.random.permutation(hist1.T).T
    hist2= np.random.permutation(hist2.T).T
    return hist1, hist2




def compute_pairwise_correlations(arrays):
    from scipy.stats import pearsonr

    n_days = len(arrays)
    n_neurons = arrays[0].shape[0]
    
    
    active_days = np.zeros(n_neurons)
    for neuron in range(n_neurons):
        for day in range(n_days):
            if np.any(arrays[day][neuron, :] > 0):
                active_days[neuron] += 1
    
    all_correlations = np.zeros((n_neurons, n_days - 1))
    for day in range(1, n_days):
        for neuron in range(n_neurons):
            trace1 = arrays[0][neuron, :]  
            trace2 = arrays[day][neuron, :]  
            try:
                corr, _ = pearsonr(trace1, trace2)
                all_correlations[neuron, day-1] = corr if not np.isnan(corr) else np.nan
            except:
                all_correlations[neuron, day-1] = np.nan
    
    mean_correlations = np.nanmean(all_correlations, axis=1)
    
    mean_correlations[active_days < 5] = np.nan
    
    return mean_correlations

def neuron_stability(datapath, Context, maps, transients=False, dim_red=False, standardize='stand',remove_inactive=False, remove_day_inactive=False):
    if remove_inactive is not False:
        _,remove_inactive = get_active_neurons_per_day(datapath,  maps, Context, standardize=standardize)
    hist = [None] * len(maps)
    for i in range(len(maps)):
        data = trial_one_session_AK(datapath, maps[i], Context, dim_red=dim_red, transients=transients, remove_inactive=remove_inactive, 
                                    standardize=standardize, remove_day_inactive=remove_day_inactive)
        hist[i] = data['hist']
    return compute_pairwise_correlations(hist)



def sort_maps_from_reference_within_session_AK(datapath, Context, maps, reference='odd', reference_type='no_reference', transients=False, dim_red=False, standardize='stand', remove_inactive=False, remove_day_inactive=False):
    hist_sorts_odd = [None] * (len(maps))
    hist_sorts_even = [None] * (len(maps))
    if remove_inactive is not False:
        _,remove_inactive = get_active_neurons_per_day(datapath, maps, Context, standardize=standardize)
    for i in range(len(maps)):
        data = trial_one_session_AK(datapath, maps[i], Context, dim_red=dim_red, transients=transients, remove_inactive=remove_inactive, standardize=standardize, remove_day_inactive=remove_day_inactive)


        transients_odd, lin_pos_odd, transients_even, lin_pos_even = get_data.separate_odd_even_runs(data['calcium'], data['pos'])
        #transients_odd, lin_pos_odd, transients_even, lin_pos_even = get_data.split_half_data(data['calcium'], data['pos'])

        hist_odd,_ = get_data.average_spatial_tuning_function(transients_odd, lin_pos_odd)
       # hist_odd = np.nan_to_num(hist_odd , nan=0.0)
        sort_odd = get_data.sort_map(hist_odd)
        hist_even,_ = get_data.average_spatial_tuning_function(transients_even, lin_pos_even)
      #  hist_even = np.nan_to_num(hist_even , nan=0.0)
        sort_even = get_data.sort_map(hist_even)

        if reference_type == 'no_reference':
            hist_sorts_odd[i] = hist_odd
            hist_sorts_even[i] = hist_even
        elif reference_type == 'own_reference':
            hist_sorts_odd[i] = hist_odd[sort_odd]
            hist_sorts_even[i] = hist_even[sort_even]
        elif reference_type == 'reference':
            if reference == 'odd':
                hist_sorts_odd[i] = hist_odd[sort_odd]
                hist_sorts_even[i] = hist_even[sort_odd]
            elif reference == 'even':
                hist_sorts_odd[i] = hist_odd[sort_even]
                hist_sorts_even[i] = hist_even[sort_even]
    return hist_sorts_odd, hist_sorts_even



def sort_maps_own_reference(datapath, Context, maps, hist='hist',dim_red = False, remove_inactive=False, standardize='stand'):
    if remove_inactive is not False:
        _,remove_inactive = get_active_neurons_per_day(datapath, remove_inactive, Context, standardize=standardize)    
    hist_sorts = [None] * (len(maps))
    for i, map in enumerate(maps):
        data = trial_one_session_AK(datapath, map, Context, dim_red=dim_red, remove_inactive=remove_inactive, standardize=standardize)
        hist_sorts[i] = data[hist]
    return hist_sorts



def populationgeometry(maps, method='angles'):
    angles = [None] * len(maps)
    norms = np.zeros(len(maps)-1)
    for i, map in enumerate(maps):
        a = pop.PopulationGeometry(map)
        if method == 'angles': 
            angles[i] = a.estimate_geometry_of_responses_population_vector2()
        elif method == 'subspace':
            angles[i] = a.subspace_angles_matrix()    
        elif method == 'subspace_matrix':
            angles[i] = a.principal_triplet_angles()

    e = 1
    for i in range(len(norms)):
        if method == 'angles':
            norms[i] =  pop.PopulationGeometry.frobenius_norm_difference(angles[0], angles[e])
        elif method == 'subspace':
            norms[i] = pop.PopulationGeometry.frobenius_norm_difference(angles[0], angles[e])
        elif method == 'subspace_matrix':
            norms[i] = pop.PopulationGeometry.norm_difference(angles[0], angles[e])

        e+=1
    return norms



def populationgeometry_context(maps1, maps2, method='angles'):
    angles1 = [None] * len(maps1)
    angles2 = [None] * len(maps2)
    norms = np.zeros(len(maps1))
    for i in range(len(maps1)):
        a = pop.PopulationGeometry(maps1[i])
        b = pop.PopulationGeometry(maps2[i])
        if method == 'angles':
            angles1[i] = a.estimate_geometry_of_responses_population_vector2()
            angles2[i] = b.estimate_geometry_of_responses_population_vector2()
        elif method == 'subspace':
            angles1[i] = a.subspace_angles_matrix()
            angles2[i] = b.subspace_angles_matrix()

        if method == 'angles':
            norms[i] =  pop.PopulationGeometry.frobenius_norm_difference(angles1[i], angles2[i])
        elif method == 'subspace':
            norms[i] = pop.PopulationGeometry.frobenius_norm_difference(angles1[i], angles2[i])

    return norms


def topology_analysis(maps, max_dim=4):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)    
        persistence_diagrams = [None] * len(maps)
        norms = np.zeros(len(maps) - 1)
        
        for i, map in enumerate(maps):
            a = pop.PopulationGeometry(map)
            persistence_diagrams[i] = a.compute_persistent_homology(max_dim)
        
        e = 1
        for i in range(len(norms)):
            dim_distances = [pop.PopulationGeometry.compare_topology(persistence_diagrams[0][d], persistence_diagrams[e][d]) 
                            for d in range(max_dim + 1)]
            norms[i] = np.mean(dim_distances)
            e += 1
        
        return norms

def topology_analysis_context(maps1, maps2, max_dim=4):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)    
        persistence_diagrams1 = [None] * len(maps1)
        persistence_diagrams2 = [None] * len(maps2)
        norms = np.zeros(len(maps1))
        for i in range(len(maps1)):
            a = pop.PopulationGeometry(maps1[i])
            persistence_diagrams1[i] = a.compute_persistent_homology(max_dim)

            b = pop.PopulationGeometry(maps2[i])
            persistence_diagrams2[i] = b.compute_persistent_homology(max_dim)
            
            dim_distances = [pop.PopulationGeometry.compare_topology(persistence_diagrams1[i][d], persistence_diagrams2[i][d]) 
                            for d in range(max_dim + 1)]
            norms[i] = np.mean(dim_distances)

        return norms


def calculate_speed(pos, time_per_frame=1):
    velocity = np.diff(pos) / time_per_frame
    speed = np.abs(velocity)
    speed = np.pad(speed, (1, 0), mode='edge')
    return speed


def scale_and_digitize(data, num_bins=10):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    digitized_data = np.digitize(scaled_data, bins=np.linspace(scaled_data.min(), scaled_data.max(), num_bins))
    return scaled_data, digitized_data

def prep_dpca(transients, pos, smooth_sigma_raw=1, smooth_sigma_avg=3.0):
    from scipy.ndimage import gaussian_filter1d
    
    def smooth_raw_traces(data, sigma):
        if sigma <= 0:
            return data
        smoothed = np.zeros_like(data)
        for n in range(data.shape[0]):
            smoothed[n] = gaussian_filter1d(data[n], sigma)
        return smoothed
    
    def smooth_position_tuning(data, sigma):
        if sigma <= 0:
            return data
        smoothed = np.zeros_like(data)
        for n in range(data.shape[0]):
            valid_mask = ~np.isnan(data[n])
            if np.sum(valid_mask) > 3: 
                valid_data = data[n][valid_mask]
                smoothed_valid = gaussian_filter1d(valid_data, sigma)
                smoothed[n][valid_mask] = smoothed_valid
            else:
                smoothed[n] = data[n]
        return smoothed

    trial_start_indices, trial_end_indices = get_data.identify_runs(pos)
    

    pos_normalized = pos.copy()
    pos_normalized = (pos_normalized - pos_normalized.min()) / (pos_normalized.max() - pos_normalized.min()) * 20
    position_labels = np.round(pos_normalized)
    position_labels[position_labels == 0] = 1
    

    n_neurons = transients.shape[0]
    n_trials = len(trial_start_indices)
    unique_positions = np.unique(position_labels)
    n_positions = len(unique_positions)
    
    X = np.zeros((n_neurons, n_positions))
    X_trials = np.zeros((n_trials, n_neurons, n_positions))
    
    if smooth_sigma_raw > 0:
        transients = smooth_raw_traces(transients, smooth_sigma_raw)
    
    # Compute trial-by-trial data
    valid_trial_counts = np.zeros((n_neurons, n_positions))
    position_sums = np.zeros((n_neurons, n_positions))
    
    for trial in range(n_trials):
        start = trial_start_indices[trial]
        end = trial_end_indices[trial]
        

        trial_data = transients[:, start:end]
        trial_positions = position_labels[start:end]
        
        for i, pos_val in enumerate(unique_positions):
            mask = trial_positions == pos_val
            if np.sum(mask) > 0:
                trial_avg = np.mean(trial_data[:, mask], axis=1)
                X_trials[trial, :, i] = trial_avg
                valid_mask = ~np.isnan(trial_avg)
                position_sums[valid_mask, i] += trial_avg[valid_mask]
                valid_trial_counts[valid_mask, i] += 1
            else:
                X_trials[trial, :, i] = np.nan
    
    # Remove trials with missing positions
    valid_trials = ~np.isnan(X_trials).any(axis=(1, 2))
    X_trials = X_trials[valid_trials]
    
    # Compute trial-averaged data
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.where(valid_trial_counts > 0, 
                    position_sums / valid_trial_counts, 
                    np.nan)
    
    if smooth_sigma_avg > 0:
        X = smooth_position_tuning(X, smooth_sigma_avg)
    
    
    return X, X_trials


def reshape_array(array):
    shape = array.shape
    transposed_array = array.transpose(1, 0, *range(2, len(shape)))
    new_shape = (shape[1], np.prod(shape[0:1] + shape[2:]))
    reshaped_array = transposed_array.reshape(new_shape)
    return reshaped_array


def _align_sets_core(calcium1, calcium2, pos1, pos2, method='dpca', n_components=10):
    from scipy.linalg import subspace_angles
    from scipy.ndimage import gaussian_filter1d
    
    def smooth_tuning_curves(hist_data, sigma):
        smoothed = np.zeros_like(hist_data)
        for neuron in range(hist_data.shape[0]):
            padded = np.hstack([hist_data[neuron, -3*sigma:], 
                            hist_data[neuron], 
                            hist_data[neuron, :3*sigma]])
        
            smoothed_padded = gaussian_filter1d(padded, sigma)
            smoothed[neuron] = smoothed_padded[3*sigma:-3*sigma]
            
        return smoothed
    
    translation_scale1 = pop.ManifoldAlignment.get_scaling_translation(calcium1.T)
    translation_scale2 = pop.ManifoldAlignment.get_scaling_translation(calcium2.T)

    calcium1_scaled = pop.ManifoldAlignment.apply_scaling_translation(calcium1.T, translation_scale1[0], translation_scale1[1]).T
    calcium2_scaled = pop.ManifoldAlignment.apply_scaling_translation(calcium2.T, translation_scale2[0], translation_scale2[1]).T

    hist1, _ = get_data.average_spatial_tuning_function(calcium1_scaled, pos1)
    hist2, _ = get_data.average_spatial_tuning_function(calcium2_scaled, pos2)

    if method == 'pca':
        from sklearn.decomposition import PCA
        pca1 = PCA(n_components=n_components)
        pca2 = PCA(n_components=n_components)

        hist1_smooth = smooth_tuning_curves(calcium1_scaled, 4)
        hist2_smooth = smooth_tuning_curves(calcium2_scaled, 4)
        X_rotations = pca1.fit(hist1_smooth.T).components_.T # shape (n_neurons, n_components)
        Y_rotations = pca2.fit(hist2_smooth.T).components_.T

        
        weights1 = np.sqrt(pca1.explained_variance_ratio_)
        weights2 = np.sqrt(pca2.explained_variance_ratio_)
    else:  #
        from dPCA import dPCA
        X1, X_trials1 = prep_dpca(calcium1_scaled, pos1)
        X2, X_trials2 = prep_dpca(calcium2_scaled, pos2)
        
        dpca1 = dPCA.dPCA(labels='p', n_components=n_components, regularizer=None)
        dpca2 = dPCA.dPCA(labels='p', n_components=n_components, regularizer=None)
        
        dpca1.protect = []
        dpca2.protect = []
        
        _ = dpca1.fit_transform(X1, trialX=X_trials1) 
        _ = dpca2.fit_transform(X2, trialX=X_trials2)
        
        X_rotations = dpca1.D['p'] 
        Y_rotations = dpca2.D['p']
        weights1 = np.sqrt(dpca1.explained_variance_ratio_['p'][:n_components])
        weights2 = np.sqrt(dpca2.explained_variance_ratio_['p'][:n_components])
    

    def weighted_procrustes(X, Y, weights1=None, weights2=None):
        if weights1 is None:
            weights1 = np.ones(X.shape[1])
        if weights2 is None:
            weights2 = np.ones(Y.shape[1])

        W1 = np.diag(weights1)
        W2 = np.diag(weights2)
        X_w = X @ W1
        Y_w = Y @ W2
        alignment = pop.ManifoldAlignment(X_w, Y_w)
        return alignment.align_manifolds()
    

    angles = np.rad2deg(subspace_angles(X_rotations, Y_rotations))
    rot_data = weighted_procrustes(X_rotations, Y_rotations, weights1, weights2)
    R = rot_data[3]

    R_high = Y_rotations @ R @ X_rotations.T
    test_rot = Y_rotations.T @ hist2

    error = np.sum(np.square(np.dot(hist2.T, R_high).T - hist1))

    # apply rotation to the original high-dimensional data
    rotated = R_high.T @ calcium2_scaled

    # project calcium1_scaled to x_rotations
    rotated = pop.ManifoldAlignment.reverse_scaling_translation(rotated.T, translation_scale1[0], translation_scale1[1]).T
    calcium1_final = pop.ManifoldAlignment.reverse_scaling_translation(calcium1_scaled.T, translation_scale1[0], translation_scale1[1]).T

    # calculate performance
    rotated_hist_norm, _ = get_data.average_spatial_tuning_function(rotated, pos2)
    hist1_final, _ = get_data.average_spatial_tuning_function(calcium1_final, pos1)
    hist_sort_map = get_data.sort_map(hist1_final)

    new_rot_data = [R, R_high, X_rotations, Y_rotations, test_rot, error, rot_data[4]]
    principal_angle = pop.ManifoldAnalysis.principal_angle(R_high)
    trace = pop.ManifoldAnalysis.trace_magnitude(R_high)
    frobenius = pop.ManifoldAnalysis.frobenius_norm_difference(R_high)

    invese_R = Y_rotations @ R.T @ X_rotations.T

    translation_scale = {
        'translation': np.linalg.norm(translation_scale2[0] - translation_scale1[0]) / np.sqrt(calcium1.shape[0]),
        'scale_factor': translation_scale2[1] / translation_scale1[1],
        'angles': angles,
        'mean_angle': np.mean(angles),
        'min_angle': np.min(angles),
        'principal_angle': principal_angle,
        'trace': trace,
        'frobenius': frobenius,
        'inverse_R': invese_R
    }

    return (
        hist1_final,
        rotated_hist_norm,
        hist_sort_map,
        new_rot_data,
        rotated.T,
        pos2,
        calcium1_final.T,
        pos1,
        translation_scale
    )



def AK_align_2_sets2(datapath, session1, session2, Context, cross_context=False, method='dpca', 
                     n_components=10, remove_stable=False, remove_stable_tresh=0.7, remove_inactive=False, indices=False):
    if cross_context == False:
        Context1 = Context2 = Context
    else:
        Context1, Context2 = 'Context2', 'Context1'

    if remove_inactive is not False:
        _, remove_inactive = get_active_neurons_per_day(datapath, [str(i) for i in np.arange(0, 13)] + ['14', '15', '17', '19'], Context)

    trial_data1 = trial_one_session_AK(datapath, session1, Context1, remove_inactive=remove_inactive, 
                                       standardize='stand', conv=False, remove_stable=remove_stable, remove_stable_tresh=remove_stable_tresh, indices=indices)
    trial_data2 = trial_one_session_AK(datapath, session2, Context2, remove_inactive=remove_inactive, 
                                       standardize='stand', conv=False, remove_stable=remove_stable, remove_stable_tresh=remove_stable_tresh, indices=indices)

    return _align_sets_core(
        trial_data1['calcium'],
        trial_data2['calcium'],
        trial_data1['pos'],
        trial_data2['pos'],
        method=method,
        n_components=n_components
    )


def AK_align_2_sets_sim(calcium1, calcium2, pos1, method='dpca', n_components=10):
    return _align_sets_core(
        calcium1,
        calcium2,
        pos1,
        pos1,
        method=method,
        n_components=n_components
    )


def simulate_drift_map_first(datapath, session, Context, days, max_drift, drift_type='circular', 
                             max_amplitude_change=0, dim_red=False, odd_even=False, remove_inactive=False):
    if remove_inactive is not False:
        _, remove_inactive = get_active_neurons_per_day(datapath, [str(i) for i in np.arange(0, days)], Context) 
    trial_data = trial_one_session_AK(datapath, session, Context, dim_red=dim_red, remove_inactive=remove_inactive)
    trial_data['calcium'] = trial_data['calcium'][np.max(trial_data['calcium'], axis=1) > 0]
    initial_map, _ = get_data.average_spatial_tuning_function(trial_data['calcium'], trial_data['pos'])

 
    drifted_maps = sim.simulate_circular_drift_on_map(initial_map, days, max_drift, max_amplitude_change)

    maps, sorts = drifted_maps, [None] * len(drifted_maps)
    for i in range(len(maps)):
        sorts[i] = get_data.sort_map(maps[i])

    odd_even_data = None


    return {'drift_data': None, 'maps': maps, 'sorts': sorts, 'pos': trial_data['pos'], 'odd_even_data': odd_even_data}


def simulate_drift(datapath, session, Context, days, max_drift=0.5, drift_type='circular',  max_remap_prob=0.3,
                    amplitude_drift_prob= 0.3, amplitude_change_scale= 0.5,dim_red=False, odd_even=False, global_seed=1,
                      active_days= ['0']+ [str(i) for i in np.arange(1, 13)] +  ['14', '15', '17', '19'], standardize='stand', 
                      remove_day_inactive=True, drift_probabilities=None, remove_inactive=False, remove_stable=False, remove_stable_tresh=0.7):
    if remove_inactive is not False:
        _,remove_inactive = get_active_neurons_per_day(datapath, active_days, Context) 
    
    trial_data = trial_one_session_AK(datapath, session, Context, dim_red=dim_red, remove_inactive=remove_inactive, 
                                      standardize=standardize, remove_day_inactive=remove_day_inactive, 
                                       remove_stable=remove_stable, remove_stable_tresh=remove_stable_tresh)
    trial_data['calcium'] = trial_data['calcium'][np.max(trial_data['calcium'], axis=1) > 0]

    if drift_type == 'circular':
        drift_data = sim.improved_simulate_drift(trial_data['calcium'], trial_data['pos'], days, max_remap_prob=max_remap_prob, 
                                                  amplitude_drift_prob= amplitude_drift_prob ,amplitude_change_scale= amplitude_change_scale,
                                                  smoothness=0.00, use_uniform_density=False, global_seed=global_seed)
    elif drift_type == 'circular_variable':
        drift_data = sim.simulate_variable_circular_drift(trial_data['calcium'], trial_data['pos'], days, max_drift)
    elif drift_type == 'rotational':
        drift_data = sim.gradual_rotation(trial_data['calcium'], days, max_drift)
    elif drift_type == 'scale':
        drift_data = sim.gradual_scaling(trial_data['calcium'], days, max_drift)
    
    
    maps, sorts = [None] * len(drift_data), [None] * len(drift_data)
    for i in range(len(maps)):

        if dim_red == 'pca':
            dim_red = pop.TransientEmbedding(drift_data[i].T, method='pca', n_components=10)
            drift_data[i] = dim_red.embedding().T

        if dim_red == 'dpca':
            dim_red = pop.TransientEmbedding(drift_data[i].T, method='dpca', n_components=10)
            drift_data[i] = dim_red.embedding(pos=trial_data['pos'])

        maps[i],_ = get_data.average_spatial_tuning_function(drift_data[i], trial_data['pos'])
        #maps[i]  = np.nan_to_num(maps[i], nan=0.0)
        sorts[i] =  get_data.sort_map(maps[i])

    odd_even_data = None
    if odd_even==True:
        hist_odd = [None] * len(drift_data)
        hist_even = [None] * len(drift_data)
        for i in range(len(drift_data)):
            transients_odd, lin_pos_odd, transients_even, lin_pos_even = get_data.separate_odd_even_runs(drift_data[i], trial_data['pos'])
            hist_odd[i],_ = get_data.average_spatial_tuning_function(transients_odd, lin_pos_odd)
            hist_even[i],_ = get_data.average_spatial_tuning_function(transients_even, lin_pos_even)

        odd_even_data = {'transients_odd':transients_odd, 'lin_pos_odd':lin_pos_odd, 'hist_even':hist_even, 'hist_odd':hist_odd}

    return {'drift_data':drift_data, 'maps':maps, 'sorts':sorts, 'pos':trial_data['pos'], 'odd_even_data':odd_even_data}



def common_elements(arrays):
    sets = set(arrays[0])
    for arr in arrays[1:]:
        sets &= set(arr)
    return np.array(list(sets))

def place_cells(datapath, session, Context, bins=20, shuffles=50, threshold=95, min_transients=50):
    trial_data = trial_one_session_AK(datapath, session, Context, transients=False)
    place_cells, spatial_info_scores = pc.identify_place_cells(trial_data['calcium'], trial_data['pos'], bins, shuffles, threshold, min_transients)
    return place_cells, spatial_info_scores

def spatial_info_cells(datapath, session, Context, percentage, bins=20, remove_inactive=False, standardize='stand', remove_day_inactive=False):
    if remove_inactive is not False:
        _,remove_inactive = get_active_neurons_per_day(datapath,  [str(i) for i in np.arange(1, 13)] +  ['14', '15', '17', '19'], Context, standardize=standardize)
    trial_data = trial_one_session_AK(datapath, session, Context, transients=False, remove_inactive=remove_inactive,
                                                               standardize=standardize, remove_day_inactive=remove_day_inactive)
    _, spatial_info_scores = pc.identify_place_cells(trial_data['calcium'], trial_data['pos'], 
                                                               bins=bins, shuffles=1, threshold=95, min_transients=0)
    treshold = np.percentile(spatial_info_scores, 90)
    indices_info = np.where(spatial_info_scores <= treshold)[0]

    rng = np.random.default_rng(0)
    n_total = len(spatial_info_scores)
    n_select = int(np.round(n_total * percentage / 100))
    
    indices_random = rng.choice(
        n_total,
        size=n_select,
        replace=False  
    )

    indices_random.sort()

    return indices_info, indices_random


def loop_sessions_2datas(func, datapath, session1, sessions2,*args):
    results = []
    for session in sessions2:
        result = func(datapath, session1, session, *args)
        results.append(result)
    return results

def loop_sessions_2datas2(func, datapath, sessions1, sessions2,*args):
    results = []
    for i in range(len(sessions2)):
        session1 = sessions1[i]
        session2 = sessions2[i]
        result = func(datapath, session1, session2, *args)
        results.append(result)
    return results


def loop_sessions_1data(datapath, session1, func, Context=None, *args):
    results1 = []
    for session in session1:
        result1 = func(datapath, session, Context, *args)
        results1.append(result1)
    return results1

def loop_sessions(func, session1, sessions2, *args):
    results = []
    for session in sessions2:
        result = func(session1, session, *args)
        results.append(result)
    return results

def loop_sessions_3(func, session1, session2, pos, *args):
    results = []
    for session in session2:
        result = func(session1,session,pos, *args)
        results.append(result)
    return results

















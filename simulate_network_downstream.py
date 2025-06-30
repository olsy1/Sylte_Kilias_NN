import numpy as np
import analysis
from scipy.special import softmax
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import random
from scipy.stats import special_ortho_group
import get_data
import population_activity as pop
import helper_functions as hf

np.random.seed(42)
random.seed(42)
noise_rng = np.random.RandomState(42)


def scale_and_digitize(data, min_val=1, max_val=20, num_bins=20):
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * (max_val - min_val) + min_val
    bins = np.linspace(min_val, max_val, num_bins + 1)
    digitized_data = np.digitize(scaled_data, bins, right=True)
    digitized_data = np.clip(digitized_data, 1, num_bins) - 1 
    return scaled_data, digitized_data


def generate_controlled_shear(n_dim, shear_factor=1.0):
    n_shear_dims = n_dim - 1
    transform = np.eye(n_dim)
    for i in range(n_shear_dims):
    #for i in range(15):
        transform[i, i+1] = shear_factor
    return transform


def generate_random_transformation(dim, n_transforms=10, sv_min=0, sv_max=5, random_state=42):
    rng = np.random.RandomState(random_state)
    transformations = []
    
    while len(transformations) < n_transforms:
        sv_noise_level = np.random.uniform(sv_min, sv_max)
        Q = special_ortho_group.rvs(dim=dim, random_state=rng)
        eps = np.random.randn(dim) * sv_noise_level
        mask = np.random.rand(dim) < 0.3  # 30% of sv perturbed
        eps *= mask
        sv = np.exp(eps)
        # scale determinant to 1
        sv = sv / (np.prod(sv) ** (1 / dim))

        U = special_ortho_group.rvs(dim=dim, random_state=rng)
        R = U @ np.diag(sv) @ U.T
        M = Q @ R
        
        if np.linalg.cond(M) <= 1e3:  
            transformations.append(M)
    
    return transformations


def measure_orthogonality(M):
    I = np.eye(M.shape[0])
    diff = M.T @ M - I
    return np.sqrt(np.linalg.norm(diff, 'fro')) / np.sqrt(np.sqrt(M.shape[0]))


class PositionDecoder:
    def __init__(self, n_positions=20, ridge_lambda=10):
        self.n_positions = n_positions
        self.ridge_lambda = ridge_lambda
        self.weights = None
        self.feature_mask = None
        
    def fit(self, X, y):
        self.feature_mask = np.var(X, axis=0) > 1e-10
        X_filtered = X[:, self.feature_mask]
        
        y_onehot = np.eye(self.n_positions)[y]
        
        n_features = X_filtered.shape[1]
        XTX = X_filtered.T @ X_filtered
        reg_matrix = self.ridge_lambda * np.eye(n_features)
        
        self.weights = np.zeros((X.shape[1], self.n_positions))
        weights_filtered = np.linalg.solve(XTX + reg_matrix, X_filtered.T @ y_onehot)
        self.weights[self.feature_mask] = weights_filtered
    
    def get_individual_decoder_outputs(self, X, noise_level=0.0):
        if noise_level > 0:
            X = X + noise_rng.normal(0, noise_level, X.shape)
        return X @ self.weights
    
    def predict_proba(self, X):
        logits = self.get_individual_decoder_outputs(X)
        return softmax(logits, axis=1)
    
    def predict(self, X, noise_level=0.0):
        if noise_level > 0:
            X = X + noise_rng.normal(0, noise_level, X.shape)
        logits = X @ self.weights
        return np.argmax(logits, axis=1)


def analyze_decoding(X, y, decoder, noise_level=0.0):
    if noise_level > 0:
        X = X + noise_rng.normal(0, noise_level, X.shape)
    
    results = {}
    y_pred = decoder.predict(X)
    results['mae'] = mean_absolute_error(y, y_pred)
    
    conf_matrix = np.zeros((decoder.n_positions, decoder.n_positions))
    for true, pred in zip(y, y_pred):
        conf_matrix[true, pred] += 1
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_matrix = conf_matrix / row_sums
    results['confusion_matrix'] = conf_matrix
    
    probs = decoder.predict_proba(X)
    results['confidence'] = {
        'mean': np.mean(np.max(probs, axis=1)),
        'std': np.std(np.max(probs, axis=1)),
        'distribution': probs
    }
    return results


def analyze_individual_decoders(X, y, decoder, noise_level=0.0, maps_downstream_pre=None):
    if noise_level > 0:
        X = X + noise_rng.normal(0, noise_level, X.shape)
    
    results = {}
    decoder_outputs = decoder.get_individual_decoder_outputs(X)

    maps_downstream_post,_ = get_data.average_spatial_tuning_function(decoder_outputs.T, y)
    drift_downstream = pop.ManifoldAnalysis.population_correlation(maps_downstream_post, maps_downstream_pre)
    results['drift_downstream'] = drift_downstream
    
    tuning_curves = np.zeros((decoder.n_positions, decoder.n_positions))
    for pos in range(decoder.n_positions):
        pos_mask = y == pos
        tuning_curves[:, pos] = np.mean(decoder_outputs[pos_mask], axis=0)
    results['tuning_curves'] = tuning_curves
    
    selectivity = np.zeros(decoder.n_positions)
    for i in range(decoder.n_positions):
        preferred_pos = np.argmax(tuning_curves[i])
        preferred_response = tuning_curves[i, preferred_pos]
        other_response = np.mean(np.delete(tuning_curves[i], preferred_pos))
        selectivity[i] = (preferred_response - other_response) / (preferred_response + other_response)
    results['selectivity'] = selectivity
    
    decoder_correlations = np.corrcoef(decoder_outputs.T)
    results['decoder_correlations'] = decoder_correlations
    
    snr = np.zeros(decoder.n_positions)
    for i in range(decoder.n_positions):
        pos_mask = y == i
        signal = np.mean(decoder_outputs[pos_mask, i])
        noise = np.std(decoder_outputs[pos_mask, i])
        snr[i] = signal / noise if noise != 0 else 0
    results['snr'] = snr
    
    return results

def simulate_shear(datapath, day, context, noise_levels=[0.0, 0.1, 0.2, 0.5, 1.0], shear_factors=[1.0]):
    data = analysis.trial_one_session_AK(datapath, day, context, standardize='stand_m', remove_day_inactive=True)
    X = data['calcium'] # neurons x time
    X = X.T
    pos = data['pos']
    _,y = scale_and_digitize(pos, num_bins=20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )
   
    decoder = PositionDecoder()
    decoder.fit(X_train, y_train)
   
    decoder_outputs_pre = decoder.get_individual_decoder_outputs(X_train)
    maps_downstream_pre, _ = get_data.average_spatial_tuning_function(decoder_outputs_pre.T, y_train)


    rotation_matrix = special_ortho_group.rvs(X_test.shape[1], random_state=42)

    results = {
        'original': {'final': {}, 'individual': {}},
        'rotation': {'final': {}, 'individual': {}}
    }
    
    for factor in shear_factors:
        results[f'controlled_{factor}'] = {'final': {}, 'individual': {}}
    results['transformations'] = {
        'rotation_matrix': rotation_matrix,
        'controlled_transforms': {}
    }
    
    for noise in noise_levels:
        # original
        results['original']['final'][noise] = analyze_decoding(
            X_test, y_test, decoder, noise
        )
        results['original']['individual'][noise] = analyze_individual_decoders(
            X_test, y_test, decoder, noise, maps_downstream_pre
        )
        
        # rotation
        X_rot = X_test @ rotation_matrix
        decoder_rot = PositionDecoder()
        decoder_rot.weights = rotation_matrix.T @ decoder.weights
        
        results['rotation']['final'][noise] = analyze_decoding(
            X_rot, y_test, decoder_rot, noise
        )
        results['rotation']['individual'][noise] = analyze_individual_decoders(
            X_rot, y_test, decoder_rot, noise, maps_downstream_pre
        )
        
        # for each shear
        for factor in shear_factors:
            controlled_transform = generate_controlled_shear(X_test.shape[1], shear_factor=factor)
            results['transformations']['controlled_transforms'][factor] = controlled_transform
            
            X_ctrl = X_test @ controlled_transform
            decoder_ctrl = PositionDecoder()
            decoder_ctrl.weights = np.linalg.inv(controlled_transform) @ decoder.weights
            
            results[f'controlled_{factor}']['final'][noise] = analyze_decoding(
                X_ctrl, y_test, decoder_ctrl, noise
            )
            results[f'controlled_{factor}']['individual'][noise] = analyze_individual_decoders(
                X_ctrl, y_test, decoder_ctrl, noise, maps_downstream_pre
            )
   
    return results

def simulate_random_transform(datapath, day, context, noise=1, n_transforms=50):
    data = analysis.trial_one_session_AK(datapath, day, context, standardize='stand', remove_day_inactive=False)
    X = data['calcium'].T
    pos = data['pos']
    _, y = scale_and_digitize(pos, num_bins=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

    decoder = PositionDecoder()
    decoder.fit(X_train, y_train)
    
    decoder_outputs_pre = decoder.get_individual_decoder_outputs(X_test)
    maps_downstream_pre, _ = get_data.average_spatial_tuning_function(decoder_outputs_pre.T, y_test)

    y_pred_original = decoder.predict(X_test, noise_level=noise)
    original_mae = mean_absolute_error(y_test, y_pred_original)

    transformations = generate_random_transformation(X_test.shape[1], n_transforms=n_transforms)

    angle_list = []
    subspace_list = []
    topology_list = []
    orthogonality_list = []
    mae_list = []
    drift_list = []
    conditioning_list = []
    drift_downstream_list = []

    maps1, _ = get_data.average_spatial_tuning_function(X_test.T, y_test)

    #shuff1, shuff2 = analysis.shuffle_two_sessions_hist(datapath, '0', '19', 'Context1', remove_day_inactive=False, remove_inactive=False)
    #hist_odd, hist_even = analysis.sort_maps_from_reference_within_session_AK(datapath, 'Context1', maps=[day],remove_inactive=False, remove_day_inactive=False)
        
    ##within= populationgeometry_context(hist_odd, hist_even, method=meth)[0]
    #shuff= populationgeometry_context([shuff1], [shuff2], method=meth)[0]

    for M in transformations:
        conditioning = np.linalg.cond(M)
        X_trans = X_test @ M
        M_inv = np.linalg.inv(M)
        decoder_trans = PositionDecoder()
        decoder_trans.weights = M_inv @ decoder.weights
        y_pred_trans = decoder_trans.predict(X_trans, noise_level=noise)
        mae_trans = mean_absolute_error(y_test, y_pred_trans)
        ortho_val = measure_orthogonality(M)
 
        maps2, _ = get_data.average_spatial_tuning_function(X_trans.T, y_test)
        angle = analysis.populationgeometry_context([maps1], [maps2], method='angles')[0]
        angle = np.sqrt(angle) 
        subspace = analysis.populationgeometry_context([maps1], [maps2], method='subspace')[0]
        subspace = np.sqrt(subspace)
        topology = analysis.topology_analysis_context([maps1], [maps2])[0]
        
        #geometry_diff = (geometry_diff-within) / (shuff-within)


        drift = pop.ManifoldAnalysis.population_correlation(maps1, maps2)

        decoder_outputs_post = decoder_trans.get_individual_decoder_outputs(X_trans, noise_level=noise)
        maps_downstream_post, _ = get_data.average_spatial_tuning_function(decoder_outputs_post.T, y_test)
        drift_downstream = pop.ManifoldAnalysis.population_correlation(maps_downstream_post, maps_downstream_pre)

        angle_list.append(angle)
        subspace_list.append(subspace)
        topology_list.append(topology)
        orthogonality_list.append(ortho_val)
        mae_list.append(mae_trans)
        drift_list.append(drift)
        conditioning_list.append(conditioning)
        drift_downstream_list.append(drift_downstream)

    results = {
        'original_mae': original_mae,
        'angle': angle_list,
        'subspace' : subspace_list,
        'topology' : topology_list,
        'orthogonality': orthogonality_list,
        'mae': mae_list,
        'drift': drift_list,
        'conditioning': conditioning_list,
        'drift_downstream': drift_downstream_list  
    }

    return results
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:43:00 2024

@author: Ole
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix
from scipy.stats import pearsonr
import analysis as analysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import get_data
from scipy.special import softmax
np.random.seed(42)
random.seed(42)


def shuffle_positions(y, seed=42):
    rng = np.random.default_rng(seed)
    y_shuffled = y.copy()
    y_shuffled = rng.permutation(y_shuffled)
    return y_shuffled

def circular_shuffle_features(X, lin_pos, seed=42):
    rng = np.random.default_rng(seed)
    run_starts, run_ends = get_data.identify_runs2(lin_pos)
    X_shuffled = X.copy()
    trial_seeds = rng.integers(0, 2**32, size=len(run_starts))

    for trial_idx, (start, end) in enumerate(zip(run_starts, run_ends)):
        trial_rng = np.random.default_rng(trial_seeds[trial_idx])
        trial_data = X_shuffled[start:end+1]
        for feature_idx in range(X.shape[1]):
            shift = trial_rng.integers(0, trial_data.shape[0])
            trial_data[:, feature_idx] = np.roll(trial_data[:, feature_idx], shift)

        X_shuffled[start:end+1] = trial_data
    
    return X_shuffled



def scale_and_digitize(data, min_val=1, max_val=20, num_bins=20):
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * (max_val - min_val) + min_val
    bins = np.linspace(min_val, max_val, num_bins + 1)
    digitized_data = np.digitize(scaled_data, bins, right=True)
    digitized_data = np.clip(digitized_data, 1, num_bins)
    return scaled_data, digitized_data





def train_model(X_train, y_train, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'svc':
        model = SVC(kernel='linear', probability=False, C=0.01, tol=0.0001, random_state=42)
    elif model_type == 'gnb':
        model = GaussianNB()
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=42, fit_intercept=False)
    elif model_type == 'position_decoder':
        model = PositionDecoder(n_positions=20, ridge_lambda=10)
    else:
        raise ValueError("Invalid model type. Choose 'linear', 'ridge', 'svc', or 'gnb'.")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, is_classifier, num_bins=20):
    y_pred = model.predict(X_test)
    
    if is_classifier:
        accuracy = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=range(1, num_bins+1))
        cm_prob = cm / cm.sum(axis=1, keepdims=True)
        return {'accuracy': accuracy, 'mae': mae, 'mse': mse, 'cm_prob': cm_prob}, y_pred
    else:
        correlation, _ = pearsonr(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return {'correlation': correlation, 'mae': mae, 'mse': mse}, y_pred


def decode_neural_activity(X, y, X_new_list, y_new_list, model_type='svc', num_bins=20, scale_X='standard', use_shuffled=False, shuffle_seed=42, shuffle_type='features'):
    is_classifier = model_type in ['svc', 'gnb']
    if is_classifier:
        _, y = scale_and_digitize(y, num_bins=num_bins)
        y_new_list = [scale_and_digitize(y_new, num_bins=num_bins)[1] for y_new in y_new_list]

    if scale_X == 'standard':
        scaler = StandardScaler()
    elif scale_X == 'minmax':
        scaler = MinMaxScaler()
    elif scale_X == 'none':
        scaler = None
    else:
        raise ValueError("Invalid scaling option. Choose 'standard', 'minmax', or 'none'.")
    
    if scaler:
        X = scaler.fit_transform(X)
        X_new_list = [scaler.transform(X_new) for X_new in X_new_list]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)
    
    if use_shuffled:
        rng = np.random.default_rng(shuffle_seed)
        train_seed = rng.integers(0, 2**32)
        test_seed = rng.integers(0, 2**32)
        new_seeds = rng.integers(0, 2**32, size=len(X_new_list))
        
        if shuffle_type == 'positions':
            y_train = shuffle_positions(y_train, seed=train_seed)
            y_test = shuffle_positions(y_test, seed=test_seed)
            y_new_list = [shuffle_positions(y_new, seed=new_seed) 
                         for y_new, new_seed in zip(y_new_list, new_seeds)]
                         
        elif shuffle_type == 'features':
            X_train = circular_shuffle_features(X_train, y_train, seed=train_seed)
            X_test = circular_shuffle_features(X_test, y_test, seed=test_seed)
            X_new_list = [circular_shuffle_features(X_new, y_new, seed=new_seed) 
                         for X_new, y_new, new_seed in zip(X_new_list, y_new_list, new_seeds)]
        else:
            raise ValueError("Invalid shuffle_type. Choose 'features' or 'positions'.")
    

    model = train_model(X_train, y_train, model_type)
    
    # evaluate on test set
    scores_test, predictions_test = evaluate_model(model, X_test, y_test, is_classifier, num_bins)

    # evaluate on new datasets
    scores_new_list = []
    predictions_new_list = []
    for X_new, y_new in zip(X_new_list, y_new_list):
        scores_new, predictions_new = evaluate_model(model, X_new, y_new, is_classifier, num_bins)
        scores_new_list.append(scores_new)
        predictions_new_list.append(predictions_new)
    
    return scores_test, scores_new_list, predictions_test, predictions_new_list, y_test, y_new_list



class PositionDecoder:
    def __init__(self, n_positions=20, ridge_lambda=0):
        self.n_positions = n_positions
        self.ridge_lambda = ridge_lambda
        self.weights = None
        self.feature_mask = None
        self.noise_rng = np.random.default_rng(42)
       
    def fit(self, X, y):

        y_zero_based = y - 1
        
        # filter out features with zero variance
        self.feature_mask = np.var(X, axis=0) > 1e-10
        X_filtered = X[:, self.feature_mask]
       
        y_onehot = np.eye(self.n_positions)[y_zero_based]
        n_features = X_filtered.shape[1]
        XTX = X_filtered.T @ X_filtered
        reg_matrix = self.ridge_lambda * np.eye(n_features)

        self.weights = np.zeros((X.shape[1], self.n_positions))
        
        # solve for filtered features only
        weights_filtered = np.linalg.solve(XTX + reg_matrix, X_filtered.T @ y_onehot)
        self.weights[self.feature_mask] = weights_filtered
   
    def get_individual_decoder_outputs(self, X, noise_level=0.0):
        if noise_level > 0:
            X = X + self.noise_rng.normal(0, noise_level, X.shape)
        return X @ self.weights
   
    def predict_proba(self, X):
        logits = self.get_individual_decoder_outputs(X)
        return softmax(logits, axis=1)
   
    def predict(self, X, noise_level=0.0):
        if noise_level > 0:
            X = X + self.noise_rng.normal(0, noise_level, X.shape)
        logits = X @ self.weights
        return np.argmax(logits, axis=1) + 1
    
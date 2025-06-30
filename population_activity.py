import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from scipy.stats import trim_mean, pearsonr, spearmanr
from numpy.linalg import norm
from scipy.spatial import procrustes
from dPCA import dPCA
import get_data
import helper_functions as hf

from ripser import ripser
from persim import wasserstein
#from persim import bottleneck


class ManifoldAlignment:
    """
    A class for aligning two manifolds using scaling, translation, and rotation.
    """
    def __init__(self, manifold1, manifold2):
        self.manifold1 = manifold1
        self.manifold2 = manifold2

    def align_manifolds(self):
        if self.manifold1.shape[0] < self.manifold2.shape[0]:
            self.manifold1, self.manifold2 = self._truncate_arrays(self.manifold1, self.manifold2)
        else:
            self.manifold2, self.manifold1 = self._truncate_arrays(self.manifold2, self.manifold1)
   
        return self._procrustes(self.manifold2, self.manifold1)
   
    def _truncate_arrays(self, short_array, long_array):
        # Truncate the longer array to match the shorter one
        return short_array, long_array[:short_array.shape[0], :]


    def _procrustes(self, a, b):
        u, s, vt = svd(np.dot(a.T, b))
        R = np.dot(u, vt)
        a_transformed = np.dot(a, R)
        error = np.sum(np.square(a_transformed - b))
        return a_transformed, b, a, R, error, s
    
    @staticmethod
    def _normalize(array):
        magnitude = np.linalg.norm(array)
        return array / magnitude
    
    def rotation(res, R): 
        return np.dot(res, R)
    
    @staticmethod
    def random_rotation(data):
        random_rot = stats.special_ortho_group.rvs(len(data.T), random_state =0)
        return np.dot(data,random_rot)

    @staticmethod
    def get_scaling_translation(res):
        translation = np.mean(res, axis=0)
        magnitude = np.linalg.norm(res-translation)
        return [translation, magnitude]

    @staticmethod
    def apply_scaling_translation(res, translation, magnitude):
        return (res - translation) / magnitude
    
    @staticmethod
    def reverse_scaling_translation(res, translation, magnitude):
        return res * magnitude + translation

    @staticmethod
    def _translate_array(array_a):
        centroid_a = np.mean(array_a, axis=0)
        return array_a - centroid_a


class ManifoldAnalysis:
    @staticmethod
    def reconstruction_error(array_a, array_b):
        return np.sum(np.square(array_a - array_b))
    
    @staticmethod
    def population_correlation(array_a, array_b):
        return np.corrcoef(array_a.flatten(), array_b.flatten())[0, 1]


    @staticmethod
    def frobenius_norm_difference(R):
        n = R.shape[0]
        I = np.eye(n)
        return np.linalg.norm(R - I, 'fro')/np.sqrt(R.shape[0])
    
    @staticmethod
    def principal_angle(R):
        n = R.shape[0]
        trace = np.trace(R)
        return np.rad2deg(np.arccos(np.clip((trace - 1) / (n - 1), -1, 1)))

    def rotation_angle_magnitude(R):
        eigvals = np.linalg.eigvals(R)  
        angles = np.angle(eigvals)  
        total_rotation_angle = np.sum(np.abs(angles))  
        return total_rotation_angle


    def geodesic_distance(R):
        from scipy.linalg import logm, norm
        R_log = logm(R) 
        distance = norm(R_log, 'fro') 
        return distance


    def trace_magnitude(R):
        matrix_trace = np.trace(R) 
        return matrix_trace / R.shape[0]
    

    def unsorted_svd(A):
        ATA = np.dot(A.T, A)
        AAT = np.dot(A, A.T)
        eigenvalues_ATA, V = np.linalg.eig(ATA)
        S = np.sqrt(np.abs(eigenvalues_ATA))
        U = np.real(U)
        V = np.real(V)
        
        for i in range(V.shape[1]):
            if np.sum(np.dot(A, V[:, i])) < 0:
                V[:, i] = -V[:, i]
        
        return U, S, V.T

    def subspace_angles_same_order(A, B):
        from scipy.linalg import orth
        Q_A = orth(A)
        Q_B = orth(B)
        M = np.dot(Q_A.T, Q_B)
        U, S, Vt = ManifoldAnalysis.unsorted_svd(M)  
        angles = np.arccos(np.clip(S, -1, 1))
        
        return angles
    
    @staticmethod
    def compute_vaf_ratio(subspace1, subspace2):
        U1, _, _ = svd(subspace1, full_matrices=False)
        U2, _, _ = svd(subspace2, full_matrices=False)
        
        numerator = np.linalg.norm(U1.T @ U2, ord='fro')**2
        denominator = min(U1.shape[1], U2.shape[1])
        
        return numerator / denominator




class PopulationGeometry:
    def __init__(self, data):
        self.data = data

    def compute_procrustes_distance(data1, data2):
        # data:neurons x timepoints
        mtx1, mtx2, disparity = procrustes(data1.T, data2.T)
        error = np.sum(np.square(mtx1 - mtx2))
        return error 


    def compute_persistent_homology(self, max_dim):
        data_transposed = self.data.T
        result = ripser(data_transposed, maxdim=max_dim, coeff=2, distance_matrix=False)
        return result['dgms']


    def estimate_geometry_of_responses_population_vector2(self):
        hist = self.data.T  # Transpose to have bins x neurons
        n_timepoints = len(hist)
        #pca = PCA(n_components=6)
        #hist = pca.fit_transform(hist)
        
        def vector_angle(v1, v2):
            v1_u = v1/np.linalg.norm(v1)
            v2_u = v2/np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        
    
        all_angles = []
        for base_point in range(n_timepoints):
            vectors = []
            for target_point in range(n_timepoints):
                if target_point != base_point:
                    v = hist[target_point] - hist[base_point]
                    vectors.append(v)
            
            vectors = np.array(vectors)
            
            n_vectors = len(vectors)
            point_angles = []
            
            for i in range(n_vectors):
                for j in range(n_vectors):
                    if i !=j:
                        angle = vector_angle(vectors[i], vectors[j])
                        point_angles.append(angle)
            
            all_angles.append(point_angles)
        
        angles_array = np.array(all_angles)
        
        return angles_array



    def subspace_angles_matrix(self, group_size=2):
        from scipy.linalg import subspace_angles
        import numpy as np
        vectors = self.data.T # transpose to positions x neurons

        n = len(vectors) - group_size + 1  # number of possible subspaces
        all_angles = []
        
        for base_point in range(n):
            base_subspace = vectors[base_point:base_point+group_size].T
            point_angles = []

            for target_point in range(n):
                if target_point != base_point:
                    target_subspace = vectors[target_point:target_point+group_size].T
                    angles = subspace_angles(base_subspace, target_subspace)
                    point_angles.extend(angles)
            
            all_angles.append(point_angles)
        
        angles_array = np.array(all_angles)
        return angles_array



    def principal_triplet_angles(self, group_size=2):
        from scipy.linalg import subspace_angles
        import numpy as np
        vectors = self.data.T  # convert to shape pos bins x neurons

        n = len(vectors)
        angles = []
        
        for i in range(n):
            for j in range(i+1):
                subspace1 = vectors[i:i+group_size].T
                subspace2 = vectors[j:j+group_size].T
                
                subspace_angles_result = subspace_angles(subspace1, subspace2)
                #subspace_angles_result = ManifoldAnalysis.compute_vaf_ratio(subspace1, subspace2)
                angles.extend(subspace_angles_result)
        
        return np.array(angles)

    @staticmethod
    def compare_topology(dgm1, dgm2):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='dgm1 has points with non-finite death times')
            distance = wasserstein(dgm1, dgm2)
            return distance

    @staticmethod
    def frobenius_norm_difference(angles1, angles2):
        return norm(angles1 - angles2, ord='fro')
    
    @staticmethod
    def norm_difference(angles1, angles2):
        return norm(angles1 - angles2)

    
    @staticmethod
    def cosine_similarity(angles1, angles2):
        from scipy.spatial.distance import cosine
        flat1, flat2 = np.array(angles1).flatten(), np.array(angles2).flatten()
        return 1 - cosine(flat1, flat2)  
    
    @staticmethod
    def l1_norm_difference(angles1, angles2):
        return norm(angles1 - angles2, ord=1)
    
    @staticmethod
    def abs_difference(angles1, angles2):
        diff = np.abs(angles1 - angles2)
        return np.sum(diff)


    


    @staticmethod
    def cosine_similarity2(angles1, angles2):
        angles1 = np.array(angles1)
        angles2 = np.array(angles2)
        

        if len(angles1) != len(angles2):
            raise ValueError("Input arrays must have the same length")
        

        return np.dot(angles1, angles2) / (norm(angles1) * norm(angles2))


class TransientEmbedding:
    def __init__(self, transients, method='pca', n_components=6):
        self.transients = transients
        self.method = method
        self.n_components = n_components
        self.embedding_model = None

    def embedding(self, pos=None):
        if self.method == 'pca':
            self.embedding_model = PCA(n_components=self.n_components)
            self.embedding_model.fit(self.transients)
            x_new = self.embedding_model.transform(self.transients)
        elif self.method == 'dpca':
            X1, X_trials1 = TransientEmbedding.prep_dpca(self.transients.T, pos)
            self.embedding_model = dPCA.dPCA(labels='p', n_components=self.n_components, regularizer=None,  random_state=1)
            self.embedding_model.protect = []
            _ = self.embedding_model.fit_transform(X1, trialX=X_trials1)
            n_neurons, n_timepoints = self.transients.T.shape
            neural_data_reshaped = self.transients.T.reshape(n_neurons, n_timepoints, 1)
            Z_full = self.embedding_model.transform(neural_data_reshaped)
            x_new= Z_full['p'][:,:,0]
        elif self.method == 'fa':
            self.embedding_model = FactorAnalysis(n_components=self.n_components)
            self.embedding_model.fit(self.transients)
            x_new = self.embedding_model.transform(self.transients)
        else:
            raise ValueError("Invalid embedding method. Choose either 'pca' or 'dpca'.")

        return x_new

    def inverse(self, x_new):
        if self.method == 'pca' or self.method == 'fa':
            return self.embedding_model.inverse_transform(x_new)
        elif self.method == 'dpca':
            self.embedding_model.D['p']
            return self.embedding_model.inverse_transform(x_new)
    
    def components(self):
        if self.method == 'pca' or self.method == 'fa':
            return self.embedding_model.components_.T
        elif self.method == 'dpca':
            return self.embedding_model.D['p']

        return self.embedding_model.components_
   
    def mean(self):
        return self.embedding_model.mean_

    def prep_dpca(transients, pos):
        trial_start_indices, trial_end_indices = get_data.identify_runs(pos)
        _, position_labels = hf.scale_and_digitize(pos)

        pos = pos/max(pos)
        pos = pos-min(pos)
        pos = pos*20
        position_labels= np.round(pos)
        position_labels[position_labels==0]=1

        n_neurons = transients.shape[0]
        n_trials = len(trial_start_indices)
        unique_positions = np.unique(position_labels)
        n_positions = len(unique_positions)

        X = np.zeros((n_neurons, n_positions))
        X_trials = np.zeros((n_trials, n_neurons, n_positions))

        for i, pos in enumerate(unique_positions):
            mask = position_labels == pos
            X[:, i] = np.mean(transients[:, mask], axis=1)

        # compute trial-by-trial data
        for trial in range(n_trials):
            start = trial_start_indices[trial]
            end = trial_end_indices[trial]
            trial_data = transients[:, start:end]
            trial_positions = position_labels[start:end]
                
            for i, pos in enumerate(unique_positions):
                mask = trial_positions == pos
                if np.sum(mask) > 0:  
                    X_trials[trial, :, i] = np.mean(trial_data[:, mask], axis=1)
                else:
                    X_trials[trial, :, i] = np.nan 

        valid_trials = ~np.isnan(X_trials).any(axis=(1,2))
        X_trials = X_trials[valid_trials]
        return X, X_trials



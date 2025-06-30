
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools
from itertools import combinations


def fit_mixed_model(data_dict, standardize=True):

    from statsmodels.formula.api import mixedlm
    from sklearn.preprocessing import StandardScaler
    
    data_dict = data_dict.copy()
    
    if 'animal_id' not in data_dict:
        animal_ids = []
        for i, x_array in enumerate(data_dict['x1']):
            animal_ids.append(np.full(len(x_array), i+1))
        data_dict['animal_id'] = animal_ids

    x_vars = [key for key in data_dict.keys() if key.startswith('x')]
    
    data_cols = {}
    for key in data_dict:
        if key in ['y', 'animal_id'] or key.startswith('x'):
            data_cols[key] = np.hstack(data_dict[key])
    
    data = pd.DataFrame(data_cols)
    
    if standardize:
        scaler = StandardScaler()
        std_vars = []
        for x_var in x_vars:
            std_name = f'{x_var}_std'
            data[std_name] = scaler.fit_transform(data[[x_var]])
            std_vars.append(std_name)
            
        data['y_std'] = StandardScaler().fit_transform(data[['y']])
        formula = "y_std ~ " + " + ".join(std_vars)
        prefix = 'std'
        re_vars = std_vars  
    else:
        formula = "y ~ " + " + ".join(x_vars)
        prefix = ''
        re_vars = x_vars 
    re_formula = "~" + " + ".join(re_vars)
    model = mixedlm(formula, data, groups=data["animal_id"])
    #model = mixedlm(formula, data, groups=data["animal_id"], re_formula=re_formula)
    results = model.fit()
    
    return results, data, prefix



def mixedlm(arrays):
    """
    mixed linear model
    """
    mouse = np.hstack([np.ones(len(arr)) * i for i, arr in enumerate(arrays)])
    values = np.hstack(arrays)

    values = (values - np.mean(values)) / np.std(values)

    df = pd.DataFrame({'mouse': mouse, 'values': values})

    md = smf.mixedlm("values ~ 1", df, groups=df["mouse"])
    mdf = md.fit(method=["lbfgs"])
    return mdf.summary()


def repeated_measures_anova_general(arrays):
    """
    repeated measures anova for any number of conditions
    """
    n_conditions = len(arrays)
    n_subjects, n_timepoints = arrays[0].shape
 
    for arr in arrays[1:]:
        assert arr.shape == (n_subjects, n_timepoints), "All arrays must have the same shape"
   
    data = []
    for condition, array in enumerate(arrays):
        for subject in range(n_subjects):
            for time in range(n_timepoints):
                data.append({
                    'subject': subject,
                    'time': time,
                    'condition': condition,
                    'value': array[subject, time]
                })
   
    df = pd.DataFrame(data)
   
    aov_results = pg.rm_anova(data=df, dv='value', within=['time', 'condition'], 
                             subject='subject', detailed=True)

    post_hoc_df = post_hoc_repeated_measures(arrays)
   
    return aov_results, post_hoc_df

def repeated_measures_anova_single_condition(array):
    """
    repeated measures anova for a single condition across time
    """
    n_subjects, n_timepoints = array.shape
    data = []
    for subject in range(n_subjects):
        for time in range(n_timepoints):
            data.append({
                'subject': subject,
                'time': time,
                'value': array[subject, time]
            })
    
    df = pd.DataFrame(data)

    aov_results = pg.rm_anova(data=df, dv='value', within=['time'], subject='subject', detailed=True)
    
    return aov_results



def two_way_repeated_measures_anova(data1, data2):
    return repeated_measures_anova_general([data1, data2])

def general_anova(arrays):
    """
    one-way ANOVA for any number of groups.
    """
    groups = [np.full(len(arr), i) for i, arr in enumerate(arrays)]
    values = np.concatenate(arrays)
    groups = np.concatenate(groups)
    df = pd.DataFrame({'group': groups, 'value': values})

    return pg.anova(data=df, dv='value', between='group', detailed=True)

def check_normality_and_equality_of_variances(arrays):
    _, p_shapiro = stats.shapiro(np.concatenate(arrays))
    _, p_levene = stats.levene(*arrays)
    
    return p_shapiro > 0.05 and p_levene > 0.05

def post_hoc_non_paired(arrays, method='tukey', p_adjust='bonferroni'):
    """
    post-hoc tests for non-paired data.
    """
    groups = [np.full(len(arr), i) for i, arr in enumerate(arrays)]
    values = np.concatenate(arrays)
    groups = np.concatenate(groups)
    df = pd.DataFrame({'group': groups, 'value': values})

    if method == 'tukey':
        return pairwise_tukeyhsd(values, groups)
    elif method in ['t-test', 'mann-whitney']:
        return pg.pairwise_tests(data=df, dv='value', between='group', 
                                 parametric=(method == 't-test'), 
                                 padjust=p_adjust)
    else:
        raise ValueError("Method must be 'tukey', 't-test', or 'mann-whitney'")


def post_hoc_repeated_measures(arrays, p_adjust='bonferroni', alpha=0.05):
    """
    post-hoc analysis for repeated measures data with automatic test selection.
    
    """
    n_conditions = len(arrays)
    n_subjects, n_timepoints = arrays[0].shape
    
    for arr in arrays[1:]:
        if arr.shape != (n_subjects, n_timepoints):
            raise ValueError("All arrays must have the same shape")

    results_list = []
    all_p_values = []
    
    for t in range(n_timepoints):
        for i, j in combinations(range(n_conditions), 2):
            data1 = arrays[i][:, t]
            data2 = arrays[j][:, t]
            
            differences = data1 - data2
            _, normality_p = stats.shapiro(differences)
            
            if normality_p > alpha:  
                statistic, p_value = stats.ttest_rel(data1, data2)
                test_name = 't-test'
            else:  
                statistic, p_value = stats.wilcoxon(data1, data2)
                test_name = 'wilcoxon'

            all_p_values.append(p_value)
            
            if test_name == 't-test':
                effect_size = np.mean(differences) / np.std(differences, ddof=1)
                effect_size_name = "Cohen's d"
            else:
                n = len(differences)
                z = np.abs(statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                effect_size = z / np.sqrt(n)
                effect_size_name = 'r'

            results_list.append({
                'Timepoint': f'T{t+1}',
                'Condition 1': f'C{i+1}',
                'Condition 2': f'C{j+1}',
                'Test Used': test_name,
                'Statistic': round(statistic, 4),
                'p-value': p_value,
                'Normal Dist.': normality_p > alpha,
                'Normality p-value': round(normality_p, 4),
                'Effect Size': round(effect_size, 4),
                'Effect Size Type': effect_size_name,
                'Mean Cond 1': round(np.mean(data1), 4),
                'Mean Cond 2': round(np.mean(data2), 4),
                'Mean Diff': round(np.mean(differences), 4),
                'SD Diff': round(np.std(differences, ddof=1), 4),
                'N': len(data1)
            })
    
    df = pd.DataFrame(results_list)
    
    adjusted_p_values = pg.multicomp(all_p_values, method=p_adjust)[1]
    df[f'Adjusted p-value ({p_adjust})'] = adjusted_p_values
    
    df['Significance'] = ''
    df.loc[df['Adjusted p-value ({})'.format(p_adjust)] <= 0.001, 'Significance'] = '***'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.001) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.01), 'Significance'] = '**'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.01) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.05), 'Significance'] = '*'
    
    for col in ['p-value', f'Adjusted p-value ({p_adjust})', 'Normality p-value']:
        df[col] = df[col].apply(lambda x: f'{x:.4e}' if x < 0.0001 else f'{x:.4f}')
    
    column_order = [
        'Timepoint', 'Condition 1', 'Condition 2',
        'Test Used', 'Statistic', 
        'p-value', f'Adjusted p-value ({p_adjust})', 'Significance',
        'Effect Size', 'Effect Size Type', 
        'Mean Cond 1', 'Mean Cond 2', 'Mean Diff', 'SD Diff',
        'N', 'Normal Dist.', 'Normality p-value'
    ]
    df = df[column_order]
    
    return df

def post_hoc_timepoints(arrays, p_adjust='bonferroni', alpha=0.05):
    """
    post-hoc tests for paired data across all possible timepoint combinations.
    """
    if len(arrays) < 2:
        raise ValueError("At least two timepoints are required for comparison")
   
    timepoint_pairs = list(itertools.combinations(range(len(arrays)), 2))

    results_list = []
    all_p_values = []
   
    for i, (t1_idx, t2_idx) in enumerate(timepoint_pairs):
        differences = arrays[t1_idx] - arrays[t2_idx]

        _, normality_p = stats.shapiro(differences)

        if normality_p > alpha: 
            statistic, p_value = stats.ttest_rel(arrays[t1_idx], arrays[t2_idx])
            test_name = 't-test'
        else:  
            statistic, p_value = stats.wilcoxon(arrays[t1_idx], arrays[t2_idx])
            test_name = 'wilcoxon'
       
        all_p_values.append(p_value)
        
        if test_name == 't-test':
            effect_size = np.mean(differences) / np.std(differences, ddof=1)
            effect_size_name = "Cohen's d"
        else:
            n = len(differences)
            z = np.abs(statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            effect_size = z / np.sqrt(n)
            effect_size_name = 'r'
        
        results_list.append({
            'Timepoint 1': f'T{t1_idx + 1}',
            'Timepoint 2': f'T{t2_idx + 1}',
            'Test Used': test_name,
            'Statistic': round(statistic, 4),
            'p-value': p_value,
            'Normal Dist.': normality_p > alpha,
            'Normality p-value': round(normality_p, 4),
            'Effect Size': round(effect_size, 4),
            'Effect Size Type': effect_size_name,
            'Mean Diff': round(np.mean(differences), 4),
            'SD Diff': round(np.std(differences, ddof=1), 4)
        })
   
    df = pd.DataFrame(results_list)
    
    adjusted_p_values = pg.multicomp(all_p_values, method=p_adjust)[1]
    df[f'Adjusted p-value ({p_adjust})'] = adjusted_p_values
    
    df['Significance'] = ''
    df.loc[df['Adjusted p-value ({})'.format(p_adjust)] <= 0.001, 'Significance'] = '***'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.001) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.01), 'Significance'] = '**'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.01) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.05), 'Significance'] = '*'
    
    for col in ['p-value', f'Adjusted p-value ({p_adjust})', 'Normality p-value']:
        df[col] = df[col].apply(lambda x: f'{x:.4e}' if x < 0.0001 else f'{x:.4f}')
    
    column_order = [
        'Timepoint 1', 'Timepoint 2', 'Test Used', 'Statistic', 
        'p-value', f'Adjusted p-value ({p_adjust})', 'Significance',
        'Effect Size', 'Effect Size Type', 'Mean Diff', 'SD Diff',
        'Normal Dist.', 'Normality p-value'
    ]
    df = df[column_order]
    
    return df
def post_hoc_paired_multiple(arrays1, arrays2, p_adjust='bonferroni', alpha=0.05):
    """
    post-hoc tests for paired data across multiple comparisons.
    """
    if len(arrays1) != len(arrays2):
        raise ValueError("arrays1 and arrays2 must have the same length")
    
    results_list = []
    all_p_values = []
    
    for i, (arr1, arr2) in enumerate(zip(arrays1, arrays2)):
        differences = arr1 - arr2
        
        _, normality_p = stats.shapiro(differences)

        if normality_p > alpha:  
            statistic, p_value = stats.ttest_rel(arr1, arr2)
            test_name = 't-test'
        else: 
            statistic, p_value = stats.wilcoxon(arr1, arr2)
            test_name = 'wilcoxon'
        
        all_p_values.append(p_value)
      
        if test_name == 't-test':
            effect_size = np.mean(differences) / np.std(differences, ddof=1)
            effect_size_name = "Cohen's d"
        else:
            n = len(differences)
            z = np.abs(statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            effect_size = z / np.sqrt(n)
            effect_size_name = 'r'
        
        results_list.append({
            'Comparison': f'Comp {i+1}',
            'Test Used': test_name,
            'Statistic': round(statistic, 4),
            'p-value': p_value,
            'Normal Dist.': normality_p > alpha,
            'Normality p-value': round(normality_p, 4),
            'Effect Size': round(effect_size, 4),
            'Effect Size Type': effect_size_name,
            'Mean Group 1': round(np.mean(arr1), 4),
            'Mean Group 2': round(np.mean(arr2), 4),
            'Mean Diff': round(np.mean(differences), 4),
            'SD Diff': round(np.std(differences, ddof=1), 4),
            'N': len(arr1)
        })
    
    df = pd.DataFrame(results_list)
    
    adjusted_p_values = pg.multicomp(all_p_values, method=p_adjust)[1]
    df[f'Adjusted p-value ({p_adjust})'] = adjusted_p_values
    
    df['Significance'] = ''
    df.loc[df['Adjusted p-value ({})'.format(p_adjust)] <= 0.001, 'Significance'] = '***'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.001) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.01), 'Significance'] = '**'
    df.loc[(df['Adjusted p-value ({})'.format(p_adjust)] > 0.01) & 
           (df['Adjusted p-value ({})'.format(p_adjust)] <= 0.05), 'Significance'] = '*'
    
    for col in ['p-value', f'Adjusted p-value ({p_adjust})', 'Normality p-value']:
        df[col] = df[col].apply(lambda x: f'{x:.4e}' if x < 0.0001 else f'{x:.4f}')
    
    column_order = [
        'Comparison', 'Test Used', 'Statistic', 
        'p-value', f'Adjusted p-value ({p_adjust})', 'Significance',
        'Effect Size', 'Effect Size Type', 
        'Mean Group 1', 'Mean Group 2', 'Mean Diff', 'SD Diff',
        'N', 'Normal Dist.', 'Normality p-value'
    ]
    df = df[column_order]
    
    return df




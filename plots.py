
import numpy as np
import matplotlib.pyplot as plt
import get_data
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy import stats
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d
from typing import List, Optional
def plot_pie(values, subplot):
    labels = ['pl', 'cg', 'm2']
    colors = ['grey', 'white', 'yellow']
    subplot.pie(values, labels=labels, colors = colors, autopct='%1.1f%%', wedgeprops={"edgecolor":"k",'linewidth': 1,  'antialiased': True})



def boxplot_with_points_and_lines(data, xlabel, subplot,title, ylim=[0, 100], figsize=[3, 5], whis=[5, 95], s=30, showfliers=False, widths=0.5, incl_noise=True, random_range=0.1):
    n_groups = data.shape[1]
    
    plot_props = {
        'whiskerprops': {'linestyle': '-', 'linewidth': 1, 'color': 'k'},
        'boxprops': {'linestyle': '-', 'linewidth': 1, 'color': 'k'},
        'medianprops': {'linestyle': '-', 'linewidth': 1, 'color': 'k'}
    }
    subplot.set_title(title)
    subplot.boxplot(data, whis=whis, showfliers=showfliers, widths=widths, **plot_props)
    
    subplot.set_xticks(range(1, n_groups + 1))
    subplot.set_xticklabels(xlabel)
    subplot.set_ylim(ylim)
    
    x_coords = np.arange(1, n_groups + 1)
    if incl_noise:
        x_coords = np.array([np.random.uniform(x - random_range, x + random_range, data.shape[0]) for x in x_coords])
    else:
        x_coords = np.tile(x_coords, (data.shape[0], 1))

    for i in range(data.shape[0]):
        subplot.plot(x_coords[:, i], data[i, :], 'grey', alpha=0.5)
    

    for i in range(n_groups):
        subplot.scatter(x_coords[i], data[:, i], c='k', s=s)

    return subplot



def dual_boxplot_with_points_and_lines(data1, data2, xlabel, title, subplot, ylim=[0, 100], whis=[5, 95], s=30, showfliers=False, widths=0.5, incl_noise=True, random_range=0.1, groups=['Dataset 1', 'Dataset 2']):
    n_groups = data1.shape[1]
    assert data1.shape == data2.shape, "data1 and data2 must have the same shape"

    plot_props = {
        'whiskerprops': {'linestyle': '-', 'linewidth': 1},
        'boxprops': {'linestyle': '-', 'linewidth': 1},
        'medianprops': {'linestyle': '-', 'linewidth': 1}
    }

    bp1 = subplot.boxplot(data1, positions=np.arange(1, n_groups + 1) - 0.2, whis=whis, showfliers=showfliers, widths=widths, **plot_props, patch_artist=True)
    bp2 = subplot.boxplot(data2, positions=np.arange(1, n_groups + 1) + 0.2, whis=whis, showfliers=showfliers, widths=widths, **plot_props, patch_artist=True)

    color1, color2 = 'lightblue', 'lightgreen'
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp1[element], color='blue')
        plt.setp(bp2[element], color='green')
    for patch in bp1['boxes']:
        patch.set_facecolor(color1)
    for patch in bp2['boxes']:
        patch.set_facecolor(color2)
 
    subplot.set_xticks(range(1, n_groups + 1))
    subplot.set_xticklabels(xlabel)
    subplot.set_ylim(ylim)

    def generate_x_coords(base_positions):
        if incl_noise:
            return np.array([np.random.uniform(x - random_range, x + random_range, data1.shape[0]) for x in base_positions])
        else:
            return np.tile(base_positions, (data1.shape[0], 1))
    x_coords1 = generate_x_coords(np.arange(0.8, n_groups + 0.8))
    x_coords2 = generate_x_coords(np.arange(1.2, n_groups + 1.2))

    for i in range(data1.shape[0]):
        subplot.plot(x_coords1[:, i], data1[i, :], 'blue', alpha=0.3, zorder=2)
        subplot.plot(x_coords2[:, i], data2[i, :], 'green', alpha=0.3, zorder=2)

    for i in range(n_groups):
        subplot.scatter(x_coords1[i], data1[:, i], c='blue', s=s, alpha=0.6, zorder=3)
        subplot.scatter(x_coords2[i], data2[:, i], c='green', s=s, alpha=0.6, zorder=3)

    subplot.legend([bp1["boxes"][0], bp2["boxes"][0]], groups, loc='upper right')
    return subplot

def boxplot1_with_points(data1,subplot, ylim = [0,100], figsize=[3,5],whis=[5,95],showfliers=False,widths=0.5,s=30,incl_noise=True,random_range=0.1,align='center',log=False):
    
    data1=np.asarray(data1)
    data1=data1[~np.isnan(data1)]

    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    data = data1
    subplot.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    subplot.set_ylim(ylim)
    x1=[]

 
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)

            
    subplot.scatter(x1,data1,c="k",s=s)

    if log==True:
        subplot.set_yscale('log')   



def boxplot2_with_points_and_lines(data1,data2,xlabel,title,  subplot, ylim = [0,100], figsize=[3,5],whis=[5,95],showfliers=False,widths=0.5,s=30,incl_noise=True,random_range=0.1,align='center',log=False):
    
    data1=np.asarray(data1)
    data2=np.asarray(data2)
    data1=data1[~np.isnan(data1)]
    data2=data2[~np.isnan(data2)]
    
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    #fig,ax=plt.subplots(figsize=figsize)   
    data=[data1,data2]
    subplot.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    subplot.set_xticks(ticks = [1,2], labels = xlabel)
    subplot.set_title(title)
    subplot.set_ylim(ylim)

    x1=[]
    x2=[]
   
    #
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data1)):
        _=subplot.plot([x1[n],x2[n]],[data1[n],data2[n]],"grey",alpha=0.5)  
    subplot.scatter(x1,data1,c="k",s=s)
    subplot.scatter(x2,data2,c="k",s=s)
    if log==True:
        subplot.set_yscale('log')
        
def boxplot3_with_points_and_lines(data1,data2,data3,xlabel, title, subplot, ylim = [0,100], figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    #fig,ax=plt.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3]
    subplot.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]
    subplot.set_xticks(ticks = [1,2,3], labels = xlabel)
    subplot.set_title(title, fontsize = 15)
    subplot.set_ylim(ylim)
    #subplot.set_ylim(40, 200)

    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data3)):
        if incl_noise==True:
            x3.append(np.random.choice(np.linspace(3-random_range,3+random_range,1000)))
        else:
            x3.append(3)

    for n in range(len(data1)):
        _=subplot.plot([x1[n],x2[n]],[data1[n],data2[n]],"grey",alpha=0.5)  
        _=subplot.plot([x2[n],x3[n]],[data2[n],data3[n]],"grey",alpha=0.5) 
    subplot.scatter(x1,data1,c="k",s=s)
    subplot.scatter(x2,data2,c="k",s=s)
    subplot.scatter(x3,data3,c="k",s=s)
    

def boxplot4_with_points_and_lines(data1,data2,data3,data4, xlabel, title, subplot, ylim = [0,100], figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    #fig,ax=plt.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3, data4]
    subplot.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]
    x4=[]
    subplot.set_xticks(ticks = [1,2,3,4], labels = xlabel)
    #subplot.set_title(title, fontsize = 15)
    #subplot.set_ylim(-0.2,0.8)
    subplot.set_ylim(ylim)
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data3)):
        if incl_noise==True:
            x3.append(np.random.choice(np.linspace(3-random_range,3+random_range,1000)))
        else:
            x3.append(3)
    for n in range(len(data4)):
        if incl_noise==True:
            x4.append(np.random.choice(np.linspace(4-random_range,4+random_range,1000)))
        else:
            x4.append(4)

    for n in range(len(data1)):
        _=subplot.plot([x1[n],x2[n]],[data1[n],data2[n]],"grey",alpha=0.5)  
        _=subplot.plot([x2[n],x3[n]],[data2[n],data3[n]],"grey",alpha=0.5) 
        _=subplot.plot([x3[n],x4[n]],[data3[n],data4[n]],"grey",alpha=0.5) 
    subplot.scatter(x1,data1,c="k",s=s)
    subplot.scatter(x2,data2,c="k",s=s)
    subplot.scatter(x3,data3,c="k",s=s)
    subplot.scatter(x4,data4,c="k",s=s)

def boxplot5_with_points_and_lines(data1,data2,data3,data4,data5, xlabel, title, subplot, figsize=[3,5],whis=[5,95],s=30,showfliers=False,widths=0.5,incl_noise=True,random_range=0.1):
    whiskerprops=dict(linestyle='-',linewidth=1,color="k")
    boxprops=dict(linestyle='-',linewidth=1,color="k")
    medianprops=dict(linestyle='-',linewidth=1,color="k")
    #fig,ax=plt.subplots(figsize=figsize)
    #_=pl.subplot(121)
    data=[data1,data2,data3, data4, data5]
    subplot.boxplot(data,whis=whis,showfliers=showfliers,boxprops=boxprops,whiskerprops=whiskerprops,medianprops=medianprops,widths=widths)
    x1=[]
    x2=[]
    x3=[]
    x4=[]
    x5=[]
    subplot.set_xticks(ticks = [1,2,3,4,5], labels = xlabel)
    #subplot.set_title(title, fontsize = 15)
    subplot.set_ylim(-0.2,1)
    #subplot.set_ylim(0, 1.1)
    
    for n in range(len(data1)):
        if incl_noise==True:
            x1.append(np.random.choice(np.linspace(1-random_range,1+random_range,1000)))
        else:            
            x1.append(1)
    for n in range(len(data2)):
        if incl_noise==True:
            x2.append(np.random.choice(np.linspace(2-random_range,2+random_range,1000)))
        else:
            x2.append(2)
    for n in range(len(data3)):
        if incl_noise==True:
            x3.append(np.random.choice(np.linspace(3-random_range,3+random_range,1000)))
        else:
            x3.append(3)
    for n in range(len(data4)):
        if incl_noise==True:
            x4.append(np.random.choice(np.linspace(4-random_range,4+random_range,1000)))
        else:
            x4.append(4)
    for n in range(len(data5)):
        if incl_noise==True:
            x5.append(np.random.choice(np.linspace(5-random_range,5+random_range,1000)))
        else:
            x5.append(5)

    for n in range(len(data1)):
        _=subplot.plot([x1[n],x2[n]],[data1[n],data2[n]],"grey",alpha=0.5)  
        _=subplot.plot([x2[n],x3[n]],[data2[n],data3[n]],"grey",alpha=0.5) 
        _=subplot.plot([x3[n],x4[n]],[data3[n],data4[n]],"grey",alpha=0.5) 
        _=subplot.plot([x4[n],x5[n]],[data4[n],data5[n]],"grey",alpha=0.5) 
    subplot.scatter(x1,data1,c="k",s=s)
    subplot.scatter(x2,data2,c="k",s=s)
    subplot.scatter(x3,data3,c="k",s=s)
    subplot.scatter(x4,data4,c="k",s=s)
    subplot.scatter(x5,data5,c="k",s=s)

def plot_polar(hist_sorts, neurons, column_labels, row_labels):
    num_conditions = len(hist_sorts)  
    fig, ax = plt.subplots(len(neurons), num_conditions, subplot_kw={'projection': 'polar'})
    for row in ax:
        for col in row:
            col.set_xticklabels([])
            col.set_yticklabels([])
    
    for condition in range(num_conditions):
        for neuron_index, neuron in enumerate(neurons):
            ax[neuron_index, condition].plot(np.linspace(0, 2*np.pi, len(hist_sorts[condition][neuron])), hist_sorts[condition][neuron])
            if neuron_index == 0:  
                ax[neuron_index, condition].set_title(column_labels[condition])
            if condition == 0:  
                ax[neuron_index, condition].set_ylabel(row_labels[neuron_index], labelpad=20)



def plot_arrays(arrays, labels, aspect='auto', xlabel=None, ylabel=None, show_colorbar=True, 
                scale=1.0, global_scale=True, remove_zero_rows=False, smooth_sigma=1.4,
                scale_per_neuron=True, cmap='RdYlBu_r'):
    
    def process_array(array):
        if remove_zero_rows:
            non_zero_mask = np.any(array != 0, axis=1)
            array = array[non_zero_mask]
        
        if smooth_sigma is not None and smooth_sigma > 0:
            pad_width = int(6 * smooth_sigma)
            padded_array = np.pad(array, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
            #array = gaussian_filter1d(array, sigma=smooth_sigma, axis=1, mode='nearest')

            array = gaussian_filter1d(padded_array, sigma=smooth_sigma, axis=1)
            array = array[:, pad_width:-pad_width]
        
        if scale_per_neuron:
            row_min = array.min(axis=1, keepdims=True)
            row_max = array.max(axis=1, keepdims=True)
            row_range = row_max - row_min
            row_range[row_range == 0] = 1
            array = (array - row_min) / row_range
        
        return array

    processed_arrays = [process_array(array) for array in arrays]
    filtered_arrays = [arr for arr in processed_arrays if arr.size > 0]
    filtered_labels = [label for arr, label in zip(processed_arrays, labels) if arr.size > 0]
    
    if not filtered_arrays:
        return None, None

    if global_scale and not scale_per_neuron:
        vmin, vmax = 0, 1
    else:
        vmin = min(np.min(arr) for arr in filtered_arrays)
        vmax = max(np.max(arr) for arr in filtered_arrays)

    max_neurons = max(array.shape[0] for array in filtered_arrays)

 
    num_arrays = len(filtered_arrays)
    fig_width = num_arrays * 2 * scale + (0.5 if show_colorbar else 0)
    fig_height = 5 * scale
    fig, axes = plt.subplots(1, num_arrays, 
                             figsize=(fig_width, fig_height),
                             constrained_layout=True)
    if num_arrays == 1:
        axes = [axes]

    ims = []
    for i, (array, label, ax) in enumerate(zip(filtered_arrays, filtered_labels, axes)):
        num_neurons = array.shape[0]
        extent = [0, array.shape[1], max_neurons, 0]  


        im = ax.imshow(array, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
        ims.append(im)
        ax.set_frame_on(False)
        ##ax.set_title(f"{label}\n({num_neurons} neurons)")
        ax.set_title(f"{label}")
        ax.set_xticks([])
        ax.set_yticks([])

        if xlabel and i == 0:
            ax.set_xlabel(xlabel)
        if ylabel and i == 0:
            ax.set_ylabel(ylabel)

    if show_colorbar:
        cbar = fig.colorbar(ims[-1], ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    else:
        plt.tight_layout()
    
    plt.show()






def plot_embedding(embedding, labels):
    plt.figure(figsize=(4, 4))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis')
    plt.colorbar()

def plot_average_geometry(all_arrays: List[np.ndarray], maps: List[str], 
                         colors: List[str], labels: List[str], 
                         directions: Optional[List[bool]] = None,
                         ax=None, figsize=(5, 4), title=None, 
                         ylim: Optional[List[float]] = None, 
                         ylabel='', xlabel='session', plot_individual: bool = True):
    
    if directions is None:
        directions = [True] * len(all_arrays)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    

    all_days = [0] + [int(m) for m in maps]
    all_days = sorted(list(set(all_days)))
    positions = []
    current_pos = 0
    
    for i, day in enumerate(all_days):
        if i == 0:
            positions.append(current_pos)
        else:
            gap = day - all_days[i-1]
            if gap > 1:
                current_pos += gap
            else:
                current_pos += 1
            positions.append(current_pos)
    
    day_to_position = {str(day): pos for day, pos in zip(all_days, positions)}
    
   
    for arr, direction, color, label in zip(all_arrays, directions, colors, labels):
        arr = np.array(arr)
        avg = np.mean(arr, axis=0)
        sem = np.std(arr, axis=0) / np.sqrt(len(arr))
        
        if direction:  
            x_pos = [day_to_position[m] for m in maps]
        else:  
            comparison_days = [str(d) for d in sorted(all_days[:-1])]  
            x_pos = [day_to_position[d] for d in comparison_days]
        
        if plot_individual:
            for i in range(len(arr)):
                ax.plot(x_pos, arr[i], '-', alpha=0.3, color=color)
        
        ax.plot(x_pos, avg, '-', color=color, label=label, linewidth=2, markersize=8)
        ax.fill_between(x_pos, avg - sem, avg + sem, alpha=0.3, color=color)
    
    if title is not None:
        ax.set_title(title)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    
    ax.set_xticks(list(day_to_position.values()))
    ax.set_xticklabels(list(day_to_position.keys()))
    ax.set_ylim(ylim)
    
    if ax.figure == plt.gcf():
        plt.tight_layout()


def plot_anova(all_arrays: List[np.ndarray], maps, colors: List[str],
                         labels: List[str], ax=None, figsize=(5, 4), title=None,
                         reverse_x: bool = True, plot_individual: bool = True,
                         ylim: Optional[List[float]] = None, ylabel='', xlabel='session'):
    
    positions = []
    current_pos = 0
    
    numeric_maps = [float(m) for m in maps]  
    
    for i, num in enumerate(numeric_maps):
        if i == 0:
            positions.append(current_pos)
        else:
            gap = num - numeric_maps[i-1]
            if gap > 1:
                current_pos += gap
            else:
                current_pos += 1
            positions.append(current_pos)
    
    x = positions if reverse_x else positions[::-1]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    for arr, color, label in zip(all_arrays, colors, labels):
        arr = np.array(arr)
        avg = np.mean(arr, axis=0)
        sem = np.std(arr, axis=0) / np.sqrt(len(arr))
        
        if plot_individual:
            for i in range(len(arr)):
                ax.plot(x, arr[i], '-', alpha=0.3, color=color)
        
        ax.plot(x, avg, '-', color=color, label=label, linewidth=2, markersize=8)
        ax.fill_between(x, avg - sem, avg + sem, alpha=0.3, color=color)

    if title is not None:
        ax.set_title(title)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    
    ax.set_xticks(x)
    ax.set_xticklabels(maps)
    ax.set_ylim(ylim)
    

    if ax.figure == plt.gcf():
        plt.tight_layout()


def plot_data(all_data, maps, reverse_x=False):
    fig, axes = plt.subplots(1, len(all_data), figsize=(10, 5), sharey=True)
    if len(all_data) == 1:
        axes = [axes]  
    
    for i, (ax, data) in enumerate(zip(axes, all_data)):
        g, g_2, simu_g, simu_g_rot, g_odd_even = data
        x = maps if not reverse_x else maps[::-1]
        
        ax.plot(x, g, 'o-', c='black', label='real data - context1')
        ax.plot(x, g_2, 'o-', c='grey', label='real data - context2')

        #ax.plot(x, simu_g, 'o-', c='red', label='sim drift: 5%. circular')
        #ax.plot(x, simu_g_rot, 'o-', c='green', label='sim drift: gradual rotation')
        # ax.plot(x, g_odd_even, 'o-', c='orange', label='real data, odd-even')

        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
        ax.set_xticks(x)
        ax.set_xticklabels(x, rotation=45)
        ax.set_title(f'Analysis for dataset {i + 1}')
        
        if i == 0: 
            ax.set_ylabel('Population geometry')
        
        ax.set_xlabel('Maps')

    plt.tight_layout()
    plt.show()



def plot_partial_residuals(results, data, prefix='std', figsize=(15, 5)):

    def partial_residuals(model, data, var):
        resid = model.resid
        coef = model.fe_params[var]
        return resid + coef * data[var]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    var_names = [f'x1_{prefix}', f'x2_{prefix}', f'x3_{prefix}'] if prefix else ['x1', 'x2', 'x3']
    
    for i, predictor in enumerate(var_names):
        pr = partial_residuals(results, data, predictor)

        sns.scatterplot(x=data[predictor], y=pr, hue=data['animal_id'], ax=axes[i])
        sns.regplot(x=data[predictor], y=pr, scatter=False, color='red', ax=axes[i], ci=68)

        x_label = f'Standardized {predictor[:-4]} (in SD units)' if prefix else f'{predictor}'
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel(f'Partial residuals + {predictor} effect')
        axes[i].set_title(f'Partial Residual Plot for {predictor}')
        axes[i].legend(title='Animal ID')

        beta = results.fe_params[predictor]
        axes[i].text(0.05, 0.95, f'β = {beta:.3f}', transform=axes[i].transAxes,
                    verticalalignment='top')
    
    plt.tight_layout()
    return fig

def plot_scatter(data, prefix='std', figsize=None):
    if prefix:
        var_names = [col for col in data.columns if col.startswith('x') and col.endswith(f'_{prefix}')]
    else:
        var_names = [col for col in data.columns if col.startswith('x') and not col.endswith('_std')]
    
    n_predictors = len(var_names)

    if figsize is None:
        figsize = (5 * n_predictors, 5)
    
    fig, axes = plt.subplots(1, n_predictors, figsize=figsize)
    
    if n_predictors == 1:
        axes = [axes]
    
    y_var = 'y_std' if prefix else 'y'
    
    for i, predictor in enumerate(var_names):
        sns.scatterplot(x=data[predictor], y=data[y_var], hue=data['animal_id'], ax=axes[i], s=100)
        sns.regplot(x=data[predictor], y=data[y_var], scatter=False, color='red', ax=axes[i], ci=68)
        
        if prefix:
            x_label = f'Standardized {predictor[:-4]} (in SD units)'
        else:
            x_label = predictor
        y_label = 'Standardized y (in SD units)' if prefix else 'y'
        
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel(y_label)

        axes[i].legend(title='Animal ID')
        

    plt.tight_layout()
    return fig


def plot_cell_runs_activity(neural_activity, positions, selected_cell, n_bins=20, threshold=1.9):
    cell_activity = neural_activity[selected_cell]
    run_starts, run_ends = get_data.identify_runs2(positions, threshold)
    n_runs = len(run_starts)
    
 
    bins = np.linspace(positions.min(), positions.max(), n_bins + 1)
    
    activity_matrix = np.zeros((n_runs, n_bins))

    for i, (start, end) in enumerate(zip(run_starts, run_ends)):
        run_positions = positions[start:end+1]
        run_activity = cell_activity[start:end+1]
        
        hist, _, _ = stats.binned_statistic(run_positions, run_activity, 
                                          statistic='mean', bins=bins)
        activity_matrix[i] = np.nan_to_num(hist, nan=0.0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(activity_matrix, aspect='auto', interpolation='nearest',
                   cmap='viridis', origin='upper')
    plt.colorbar(im, ax=ax, label='Mean activity')
    
    ax.set_xlabel('Position bins')
    ax.set_ylabel('Run number')
    ax.set_title(f'Cell {selected_cell} activity across runs')
    plt.tight_layout()
    
    return fig, ax, activity_matrix
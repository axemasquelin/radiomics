# coding: utf-8
'''
""" MIT License """

    Axel Masquelin & Thayer Alshaabi
    ---
    Description:
    ---
    ---
    Copyright (c) 2018
'''
# Dependencies
# ---------------------------------------------------------------------------- #
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.manifold import TSNE
from sklearn.metrics import auc

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import glob, os, sys
# ---------------------------------------------------------------------------- #

def dataread(filename, clean = False):
    '''
    Resample Dataset such that it is distributed
    '''
    df = pd.read_csv(
        os.path.join(os.path.split(os.getcwd())[0],
        'dataset', filename), low_memory= False)
    
    if clean:
        # Drop out the columns that have over 90% missing data points.
        df = df.dropna(thresh=len(df)*.90, axis=1)

        # Drop out any rows that are missing more than 50% of the required columns
        df = df.dropna(thresh=df.shape[1]*.5)

        # Drop out any rows that are missing the target value 
        df = df.dropna(axis=1, subset=df['ca'])

        # Fill the rest of missing entries with the mean of each col
        df = df.apply(lambda col: col.fillna(col.mean()))

    return df

def resample_df(seed, df, method):
    '''
        Equalize the number of samples in each class:
        -- method = 1 - upsample the positive cases
        -- method = 2 - downsample the negative cases
        -- method = 3 - return all cases
    '''
    df_neg = df[df.ca==0]
    df_pos = df[df.ca==1]


    # Upsample the pos samples
    if method == 1:
        df_pos_upsampled = resample(
            df_pos,
            n_samples=len(df_neg),
            replace=True, 
            random_state= seed
        )
        return pd.concat([df_pos_upsampled, df_neg])

    # Downsample the neg samples
    elif method == 2:
        df_neg_downsampled = resample(
            df_neg,
            n_samples=len(df_pos),
            replace=True, 
            random_state= seed
        )
        return pd.concat([df_pos, df_neg_downsampled])
    
    # Return Raw Dataset
    elif method == 3:
        return pd.concat([df_pos, df_neg])

    else:
        print('Error: unknown method')

def load_data(
    filename, 
    clean=False, 
    normalize=False,
    resample=2,
    seeder = 1,
    ):
    '''
        Returns:
            X: a matrix of features
            y: a target vector (ground-truth labels)
        normalize   - Option to normalize all features in the dataframe 
        clean       - Option to filter out empty data entires
        resample    - Method to use to resample the dataset
    '''

    df = resample_df(
        seeder,
        dataread(filename, clean),
        method=resample,
        
    ) 
    
    features = list(df.columns.values)                          # Converting Dataframe to list
    features.remove('Unnamed: 0')                               # Removing Indices from Feature List 
    features.remove('pid')                                      # Removing Patient ID from Feature List
    features.remove('ca')                                       # Removing Targets from Feature List

    if normalize == True:                                       # Normalize Check
        X = StandardScaler().fit_transform(df[features])        # Standardizing Features by removing mean and scaling unit (Z = (x-u)/s)
    
    else:
        X = df[features].to_numpy()
    
    target = df.ca.values                                       # Defining Targets

    return X, target, features

def plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc, reps):
    ''' Plot roc curve per fold and mean/std score of all runs '''

    plt.figure(reps, figsize=(10,8))

    plt.plot(
        mean_fpr, mean_tpr, color='k',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc)
    )

    # plot std
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        color='grey', alpha=.2, label=r'$\pm$ std.'
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('ROC Curve', fontsize=20)
    plt.legend(loc="lower right", fontsize=14)

def tsne_plot(files, r, comp, iters):
    x, y, features = load_data(                                   # Data Load Function Call
                                    filename = files + '.csv',       # Defining Filename
                                    clean=False,                        # Cleaning Data Option
                                    normalize=True,                     # Normalizing Data
                                    resample= 3,                        # Resampling Option - (1) for Upsampling (2) for Downsampling
                                    seeder = r                          # Defining Seed Value
                                    )

    tsne = TSNE(n_components = comp, verbose = 1, perplexity = 25, n_iter = iters)  # Defining t-sne Function and Variables
    tsne_results = tsne.fit_transform(x)                                            # t-sne Fit
    df_sne = pd.DataFrame()                                                         # Creating new dataframe
    df_sne['y'] = y                                                                 # Defining Classification in Dataframe
    
    if comp == 3:                                                                   # If 3 Component t-sne is selected
            
        df_sne['tsne-3d-one'] = tsne_results[:,0]                                   # Defining First t-sne dimension
        df_sne['tsne-3d-two'] = tsne_results[:,1]                                   # Defining second t-sne dimension
        df_sne['tsne-3d-three'] = tsne_results[:,2]                                 # Defining third t-sne dimension

        df_negs = df_sne.loc[df_sne['y'] == 0]
        df_pos = df_sne.loc[df_sne['y'] == 1]

        fig = plt.figure()                                                          # Creating Figure number
        ax = Axes3D(fig)                                                            # Creating 3D Figure
        ax.scatter(
            df_negs['tsne-3d-one'],                                                  # x-coordinate
            df_negs['tsne-3d-three'],                                                # z-cordinate            
            df_negs['tsne-3d-two'],                                                  # y-coordinate
            )
        
        ax.scatter(
            df_pos['tsne-3d-one'],                                                  # x-coordinate
            df_pos['tsne-3d-two'],                                                  # y-coordinate
            df_pos['tsne-3d-three'],                                                # z-cordinate            
            )

        ax.set_xlabel('t_SNE 1')
        ax.set_ylabel('t_SNE 2')
        ax.set_zlabel('t_SNE 3')

    else:                                                                           # For 2 Component t-sne
        df_sne['tsne-2d-one'] = tsne_results[:,0]                                   # Defining component one
        df_sne['tsne-2d-two'] = tsne_results[:,1]                                   # Defining component two
        
        plt.figure(figsize=(16,10))                                                 # Creating figure and defining plot size
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",                                       # Ploting component one and two
            hue="y",                                                                # defining y labels
            palette=sns.color_palette("hls", 2),                                    # color palette
            data=df_sne,                                                            # dataset location
            legend="full",                                                          # legend type
            alpha=0.6                                                               # visibility
        )
    
    plt.title("Sample Origin")
    plt.show()

def calc_auc(fps, tps, reps, plot_roc=False):
    ''' Calculate mean ROC/AUC for a given set of 
        true positives (tps) & false positives (fps) '''

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    for itr, (_fp, _tp) in enumerate(zip(fps, tps)):
        tprs.append(np.interp(mean_fpr, _fp, _tp))
        tprs[-1][0] = 0.0
        roc_auc = auc(_fp, _tp)
        aucs.append(roc_auc)

        if plot_roc:
            plt.figure(reps, figsize=(10,8))
            plt.plot(
                _fp, _tp, lw=1, alpha=0.5,
                label='ROC fold %d (AUC = %0.2f)' % (itr+1, roc_auc)
            )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if plot_roc:
        plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc, reps)
        
    return aucs

def csv_save(dataset, method, matr, dat_type = None):
    ''' Save scores to a csv file '''
    dataset = dataset.split('_')[1]
    cols = ['Reps'+str(i+1) for i in range(matr.shape[1])]
    logs = pd.DataFrame(matr, columns=cols) 
    result_dir = os.path.split(os.getcwd())[0] + "/results/"
    pth_to_save =  result_dir + dataset + "/" + dataset + '_' + method +  "_" + dat_type + ".csv"
    logs.to_csv(pth_to_save)

    print(logs)
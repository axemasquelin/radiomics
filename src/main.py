# coding: utf-8 
'''
""" MIT License """
    Authors:
        1. Axel Masquelin
        2. Thayer Alshaabi
    Description: 
    ---

    Copyright (c) 2018
'''

### Libraries
#--------------------------------------------
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os

import utils
import classifiers

#--------------------------------------------

def progressinfo(dataset, method, cur_iter, max_iter, cur_rep, max_rep):
    """
    Definition: Provides information on methodology, dataset,
    rep count, and iteration count at a given time.
    Inputs:
        1. dataset    - str value of dataset currently being used
        2. method     - str value showing machine learning methodology
        3. cur_iter   - int value showing current K-fold iteration
        4. max_iter   - int value showing max K-fold iteration
        5. cur_rep    - int value showing repetition of K-fold
        6. max_rep    - int value showing max repetition value
    """

    sys.stdout.write(
        "\r Method ({0}) Dataset ({1}), iter: {2} of {3}, rep: {4} of {5}".format(
            method, dataset, cur_iter, max_iter, cur_rep, max_rep))
    


if __name__ == '__main__':
    """
    Definition: Main Function for Radiomic Project
    To Do List:
        1. Make which methodology is run argument based for user to select
        2. Make dataset arg base for user to input (run all if no arg)
    """

    # Variable Definition
    seed = 2018             # Random Seed Value
    n_splits = 10           # Define the max number of k-fold cross validation
    reps = 10               # Defiine the max number of iterations

    tsne_check = False      # Tsne Flag to generate T-SNE Plot
    tsne_components = 2     # Number of Components for T-SNE Plot
    tsne_iters = 400        # Number of iterations for T-SNE Plot

    methods = [
                'Lasso',
                'svm',
                'rf',
                'LungRad_Cutoffs'
    ]
    
    datasets = [
                'data_0i',             # Tumor-Specific Dataset
                'data_10b',            # 10mm Band Dataset
                'data_10ib',           # 10mm Band + Tumor Dataset
                'data_15b',            # 15mm Band Dataset
                'data_15ib',           # 15mm Band + Tumor Dataset
                'data_LungRads',       # Maximum Nodule Diameter Dataset
                # 'data_Original'      # Original Dataset contains 268 Features
    ]

    for dataset in datasets:

        for method in methods:
            auc_scores = np.zeros((reps, n_splits))
            sens_scores = np.zeros((reps, n_splits))
            specs_scores = np.zeros((reps, n_splits))
            methodFig = 0     


            # Load Dataset
            if method == 'LungRad_Cutoffs':
                dataset = 'data_LungRads'

                X, y, features = utils.load_data(       # Data Load Function Call
                filename = dataset + '.csv' ,       # Defining Filename to Load
                clean = True,                       # Clean Data Flag (True, False)
                normalize = False,                   # Normalize Data Flag (True, False)
                resample = 2,                       # Resampling Options: (1) Upsampling, (2) Downsampling
                seeder = seed                       # Seed Value
            )

            else:
                X, y, features = utils.load_data(       # Data Load Function Call
                    filename = dataset + '.csv' ,       # Defining Filename to Load
                    clean = True,                       # Clean Data Flag (True, False)
                    normalize = True,                   # Normalize Data Flag (True, False)
                    resample = 2,                       # Resampling Options: (1) Upsampling, (2) Downsampling
                    seeder = seed                       # Seed Value
                )

            feat_imp = pd.DataFrame(data = features, columns = ['Feature Name'])
            rep_index = 0

            '''Repetition Loop'''
            for r in range(reps):

                if tsne_check:
                    utils.tsne_plot(dataset, r, tsne_components, tsne_iters)
                
                # Stratified K-Fold
                cv = StratifiedKFold(
                    n_splits = n_splits,    # Number of K-Fold Splits
                    shuffle = True,         # Shuffle Dataset Option
                    random_state = r        # Set Fix Seed
                ).split(X,y)            

                fprs, tprs = [], []         # Empty List of True and False Positivies
                senst, specf = [], []       # Empty List for Sensitivity and Specificity

                nfigure = 1

                for itr, (train,test) in enumerate(cv):
                    rep_index += 1
                    progressinfo(dataset, method,
                                cur_iter= itr, max_iter= n_splits,
                                cur_rep= r, max_rep= reps
                                )
                    if method != "LungRad_Cutoffs":
                        fp, tp, sens, specs, imp = classifiers.eval(    # Classifier Function Call
                                    X[train], X[test],                 # Defining X_train and X_test
                                    y[train], y[test],                 # Defining Y_train and Y_test
                                    clf=method,                        # Method Selected
                                    rep_index = rep_index)

                        feat_imp = pd.concat([feat_imp, imp], axis=1)
                        tprs.append(tp)  # Appending True Positive to Empty List
                        fprs.append(fp)  # Appending False Positives to Empty List

                    else:
                        sens, specs, acc = classifiers.LungRADS_rules(X[test], y[test])

                    nfigure += 1                            # Updating Figure Counter

                    senst.append(sens)                      # Appending Sensitivity Score to Empty List
                    specf.append(specs)                     # Appending Specificity Score to Empty List

                print('Repetition ' + str(r) + ' Done!')    # Printing Statues Update
                print("Sensitivity:")
                print(senst)
                sens_scores[r,:] = np.mean(senst)
                specs_scores[r,:] = np.mean(specf)
                result_dir = os.path.split(os.getcwd())[0] + "/results/"
                if method != "LungRad_Cutoffs":
                    auc_scores[r, :] = utils.calc_auc(fprs, tprs, nfigure, plot_roc=True)                               # Calculating AUC Scores

                    plt.savefig(result_dir + dataset.split('_')[1] + "/images/" + method + "-AUC" + "_reps" + str(r))  # Saving AUC Figure
                    plt.close()                                                                                         # Closing Figure
                    methodFig += 1                                                                                      # Updating Method Figure Counter

                print('-'*75)                # Printing Line Break

            feat_imp['mean'] = feat_imp.mean(axis=1)                                        # Mean of Feature Importance
            feat_imp['std'] = feat_imp.std(axis=1)                                          # Standard Deviation of Feature Importance
            classifiers.f_importances(feat_imp, dataset= dataset, method= method)           # Plot and Save Feature Importance to csv

            if method != "LungRad_Cutoffs":
                utils.csv_save(dataset, method, auc_scores, dat_type = 'AUC')               # Saving AUC Matrix to CSV File

            utils.csv_save(dataset, method, sens_scores, dat_type = "Sensitivity")          # Saving Sensitivity Matrix to CSV File
            utils.csv_save(dataset, method, specs_scores, dat_type = "Specificity")         # Saving Specificity Matrix to CSV File

    print('Done')

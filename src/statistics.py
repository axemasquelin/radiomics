# coding: utf-8
""" MIT License """
'''
""" MIT License """

    Axel Masquelin & Thayer Alshaabi
    ---
    Description:
    ---
    ---
    Copyright (c) 2018
'''
# Libraries
# ---------------------------------------------------------------------------- #
from unittest import result

import statsmodels.stats.multitest as smt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import string
import scipy
import sys, os, csv
# ---------------------------------------------------------------------------- #

def progressInfo(method, metric, model):
        
    sys.stdout.write(
            "\r Method: {0}, Model: {1}, Metric: {2}\n".format(method, model, metric)
    )   

def multitest_stats(df):
    """
    Definition:
    Inputs:
    Outputs:
    """
    
    pvals = np.zeros(6)         # Need to change this to be combination equation C(n,r) = (n!)/(r!(n-r)!)
    tvals = np.zeros(6)
    
    t, p = scipy.stats.ttest_ind(df[df.columns[0]], df[df.columns[1]])
    pvals[0], tvals[0] = p, t
    t, p = scipy.stats.ttest_ind(df[df.columns[0]], df[df.columns[2]])
    pvals[1], tvals[1] = p, t
    t, p = scipy.stats.ttest_ind(df[df.columns[1]], df[df.columns[2]])
    pvals[2], tvals[2] = p, t
    t, p = scipy.stats.ttest_ind(df[df.columns[0]], df[df.columns[3]])
    pvals[3], tvals[3] = p, t
    t, p = scipy.stats.ttest_ind(df[df.columns[1]], df[df.columns[3]])
    pvals[4], tvals[4] = p, t
    t, p = scipy.stats.ttest_ind(df[df.columns[2]], df[df.columns[3]])
    pvals[5], tvals[5] = p, t
  
    y = smt.multipletests(pvals, alpha=0.01, method='b', is_sorted = False, returnsorted = False)
  
    print("%s - %s | P-value: %.12f"%(df.columns[0], df.columns[1], y[1][0]))    
    print("%s - %s | P-value: %.12f"%(df.columns[0], df.columns[2], y[1][1]))    
    print("%s - %s | P-value: %.12f"%(df.columns[1], df.columns[2], y[1][2]))    
    print("%s - %s | P-value: %.12f"%(df.columns[0], df.columns[3], y[1][3])) 
    print("%s - %s | P-value: %.12f"%(df.columns[1], df.columns[3], y[1][4]))
    print("%s - %s | P-value: %.12f"%(df.columns[2], df.columns[3], y[1][5]))        
    return y

def annotatefig(sig, x1, x2, y, h):
    
    if sig < 0.05:
        if (sig < 0.05 and sig > 0.01):
            sigtext = '*'
        elif (sig < 0.01 and sig > 0.001): 
            sigtext = '**'
        elif sig < 0.001: 
            sigtext = '***'

        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.1, c='k')
        plt.text((x1+x2)*.5, y+h, sigtext , ha='center', va='bottom', color='k', fontsize = 'small')

def violin_plots(df, metric, method, key,
                 sig1 = None, sig2 = None, sig3 = None,
                 sig4 = None, sig5 = None, sig6 = None):
    """
    Definitions:
    Inputs:
    Outputs:
    """

    colors = {
                'Individual':   "BuGn",
                'Combined':     "RdBu",
             }

    titles = {
            'rf': 'Random Forest',
            'svm': 'Support Vector Machine',
            'Lasso': 'LASSO Regression'
            }

    fig, ax = plt.subplots()
    chart = sns.violinplot(data = df, cut = 0,
                           inner="quartile", fontsize = 16,
                           palette= sns.color_palette(colors[key], 8))

    chart.set_xticklabels(chart.get_xticklabels(), rotation=25, horizontalalignment='right')
    plt.xlabel("Dataset", fontsize = 14)

    if metric == 'AUC':
        plt.title(titles[method] + " " + metric.upper() + " Distribution", fontweight="bold", pad = 20)
        plt.yticks([0.5,0.6,.7,.8,.9,1], fontsize=15)
        plt.ylabel(metric.upper(), fontsize = 16)
        plt.ylim(0.45,1.07)
    else:
        plt.title(method + " " + metric.capitalize() + " Distribution", fontweight="bold")
        plt.ylabel(metric.capitalize(), fontsize = 16)
        
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if sig1 != None:
        x1, x2 = 0, 1                           # 0i vs. 10b/10ib
        y, h, col = .985, .0025, 'k'
        annotatefig(sig1, x1, x2, y, h)

    if sig2 != None:
        x1, x2 = 0, 2                           # 0i vs. 15b/15ib
        y, h, col = 1.015, .0025, 'k'
        annotatefig(sig2, x1, x2, y, h)

    if sig3 != None:
        x1, x2 = 1, 2                            # 10b/10ib vs 15b/15ib
        y, h, col = 0.993, .0025, 'k'
        annotatefig(sig3, x1, x2, y, h)

    if sig4 != None:
        x1, x2 = 0, 3                           # 0i vs. Maximum Diameter
        y, h, col = 1.065, .0025, 'k'
        annotatefig(sig4, x1, x2, y, h)

    if sig5 != None:
        x1, x2 = 1, 3                           # 10b/10ib vs Maximum Diameter
        y, h, col = 1.04, .0025, 'k'
        annotatefig(sig5, x1, x2, y, h)

    if sig6 != None:
        x1, x2 = 2, 3                           # 15b/15ib vs. Maximum Diameter
        y, h, col = .981, .0025, 'k'
        annotatefig(sig6, x1, x2, y, h)

    result_dir = os.path.split(os.getcwd())[0] + "/results/"
    if key == 'individual':
        plt.savefig(result_dir + metric + "_" + method + "_Across_" + key + "_Dataset.png", bbox_inches='tight',
                    dpi=600)
    else:
        plt.savefig(result_dir + metric + "_" + method + "_Across_" + key + "_Dataset.png", bbox_inches='tight',
                    dpi=600)

    plt.close()

if __name__ == '__main__':
    """
    Definition:
    Inputs:
    Outputs:
    ToDo:
        - Generate Table that shows mean_metric, std_metric, and p_value between datasets
    """

    methods = ['rf',
               'svm',
               'Lasso'
               ]

    models = {
             'Individual': ['0i', '10b', '15b', 'LungRads'],
             'Combined': ['0i', '10ib', '15ib', 'LungRads'],
             }

    col_names = {
                'Individual': ['Tumor', '10mm Band', '15mm Band', 'Tumor Size'],
                'Combined': ['Tumor', '10mm Band + Tumor', '15mm Band + Tumor', 'Tumor Size']
                }

    metrics = [                     # Select Metric to Evaluate
            'AUC',                  # Area under the Curve
            # 'Sensitivity',         # Network Senstivity
            # 'Specificity',         # Network Specificity         
            ]
    
    # Variable Flags
    create_violin = True
    check_stats = True

    for metric in metrics:
        print(metric)
        for method in methods:
            for key, value in models.items():
                progressInfo(method, metric, key)
                df = pd.DataFrame()                # General Dataframe to generate Bar-graph data
                np_0i = np.zeros((5,5))           # Conv Layer Dataframe for violin plots
                np_10b = np.zeros((5,5))          # Wavelet Layer Dataframe for violin plots
                np_15b = np.zeros((5,5))          # Multi-level Wavelet Dataframe

                for root, dirs, files in os.walk(os.path.split(os.getcwd())[0] + "/results/", topdown = True):
                    for name in files:
                        if (name.endswith(metric + ".csv")):
                            if (len(name.split('_')) == 3):
                                name_mod, name_meth, name_metr = name.split('_')
                                if  ((name_mod in models[key]) and (name_meth == method)):
                                    mean_ = []
                                    filename = os.path.join(root,name)
                                    with open(filename, 'r') as f:
                                        reader = csv.reader(f)
                                        next(reader)
                                        for row in reader:
                                            for l in range(len(row)-1):
                                                mean_.append(float(row[l+1]))
                                
                                    df[name_mod] = np.transpose(mean_)

                                    if metric == 'AUC':
                                        if (name_mod == models[key][0]):
                                            np_01 = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                                        if (name_mod == models[key][1]):
                                            np_10b = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                                        if (name_mod == models[key][2]):
                                            np_15b = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                                        if (name_mod == models[key][3]):
                                            np_LR = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)

                cols = df.columns.tolist()
                df.columns = col_names[key]
                
                if check_stats:
                    sig = multitest_stats(df)
                    
                if create_violin:
                    print("Violin Plots")
                    if (check_stats and metric == 'AUC'):
                        violin_plots(df, metric, method, key,
                                     sig1 = sig[1][0], sig2 = sig[1][1], sig3 = sig[1][2],
                                     sig4 = sig[1][3], sig5 = sig[1][4], sig6 = sig[1][5],
                                     )
                    else:
                        violin_plots(df, metric, method, key)
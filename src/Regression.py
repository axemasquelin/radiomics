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
# Libraries
# ---------------------------------------------------------------------------- #
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.utils import resample
from sklearn.metrics import auc, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import glob, os, sys, utils
# ---------------------------------------------------------------------------- #

def LassoCrossVal(X, y, n_splits, seed):
    #Variable Init
    alphas = np.arange(0, 0.1, 0.00001)
    avgMSE = []

    for l in range(len(alphas)):
        cv = StratifiedKFold(n_splits=n_splits,
                        shuffle=True,
                        random_state= seed
                        ).split(X, y)    
                
        meanSq = np.zeros((len(alphas), n_splits))

        for itr, (train, test) in enumerate(cv):    
                    # print('Alpha Value: ' + str(alphas[l]))
            
                if reg_method[m] == 'Lasso':

                    lassoReg = Lasso(alpha=alphas[l] ,normalize=True)
                    lassoReg.fit(X[train],y[train])
                    train_score = lassoReg.score(X[train], y[train])
                    test_score = lassoReg.score(X[test],y[test])
                    coeff_used = np.sum(lassoReg.coef_!=0)
                    y_pred = lassoReg.predict(X[test])
                    meanSq[l][itr] = (mean_squared_error(y[test], y_pred))
                    
                    
                    print('Training Score: ' + str(train_score))
                    print('Test Score: ' + str(test_score))
                    print('Mean Square Error', str(meanSq[l][itr]))
                    print('Number of Feature Used: ' + str(coeff_used) + '\n')
                    
                elif reg_method[m] == 'Ridge':
                    print('Ridge')
                
                elif reg_method[m] == 'Elastic':
                    print('Elastic')
                
        avgMSE.append(np.mean(meanSq[l]))

    print(avgMSE[np.argmin(avgMSE)])
    print(alphas[np.argmin(avgMSE)])                  

    return alphas[np.argmin(avgMSE)]

def dataread(filename, clean = False):
    '''
    Resample Dataset such that it is distributed
    '''
    df = pd.read_csv(
        os.path.join(os.path.abspath(os.pardir),
        'src/dataset', filename), low_memory= False)
    
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

def save_dataset(filename,
    features,
    method = None, 
    clean=False, 
    normalize=False,
    resample=2,
    seeder = 1,
):
    paths = r'src/dataset'
    df =dataread(filename, clean)

    df2 = pd.DataFrame()

    df2['pid'] = df['pid']
    df2['ca'] = df['ca']

    for l in range(len(features)):
        if features[l] == 'Unnamed: 0':
            print('skip')
        else:
            df2[features[l]] = df[features[l]]
    
    df2.to_csv(os.path.join(os.path.abspath(os.pardir),
        'src/dataset/data_' + str(method) + '.csv'))



if __name__ == '__main__':
    # Variable Initialization
    n_splits = 5
    seed = 2019

    # Dataset Variables
    files = ['data_Original']
    reg_method = [#'0i',             # Tumor Only Radiomics    
                  #'n0i',            # Non-Tumor Only Radiomics
                  #'10bc',           # 10 mm Boundary and Centroid Radiomics
                  #'15bc',           # 15 mm Boundary and Centroid Radiomics
                  #'Lasso',          # Lasso Regression on 268 Features
                  'LungRads',       # Lung Rads Standard
                  ]          
    
    for n in range(len(files)):

        print('Dataset: ' + str(files[n].split('_')[1]))

        X, y, features = utils.load_data(
                    filename = files[n] + '.csv',
                    clean=False,
                    normalize=True,
                    resample=2, # (2) to downsample the negative cases
                    seeder = seed
                    )

        for m in range(len(reg_method)):

            if reg_method[m] == 'Lasso':
                alpha = LassoCrossVal(X, y, n_splits, seed)

                cv = StratifiedKFold(n_splits=n_splits,
                            shuffle=True,
                            random_state= seed
                            ).split(X, y)    
                
                for itr, (train, test) in enumerate(cv):    
                
                    lassoReg = Lasso(alpha=alpha ,normalize=True)
                    lassoReg.fit(X[train],y[train])
                    coeff_used = np.sum(lassoReg.coef_!=0)
                
                    
                    print('Number of Feature Used: ' + str(coeff_used))
    
                    # Finding Important Features
                    plt.figure(itr)
                    coef = pd.Series(lassoReg.coef_,features).sort_values()
                    coef = coef[coef !=0]
                    print(coef)
                    coef.plot(kind = 'bar', title = 'Modal Coefficients')
                    #plt.savefig()
                    coef = coef.keys()
                    
            elif reg_method[m] == '0i':
                Oi_pd = pd.DataFrame()
                coef = []
                for l in range(len(features)):
                    ft = features[l][len(features[l])-2:len(features[l])]
                    if (ft == '0i'):
                        coef.append(features[l])
                
                print('Number of 0i Features: ' + str(len(coef)))
            
            elif reg_method[m] == 'n0i':
                Oi_pd = pd.DataFrame()
                coef = []
                for l in range(len(features)):
                    ft = features[l][len(features[l])-2:len(features[l])]
                    if ((ft != '0i') and (ft != '0c') and (ft != '5c')):
                        coef.append(features[l])
                
                print('Number of n0i Features: ' + str(len(coef)))
                
            elif reg_method[m] == '10b':
                nOi_pd = pd.DataFrame()
                coef = []
                for l in range(len(features)):
                    ft = features[l][len(features[l])-3:len(features[l])]
                    if (ft == '0b'):
                        print(ft)
                        coef.append(features[l])
                
                print('Number of 10b Features: ' + str(len(coef)))   
            
            elif reg_method[m] == '15b':
                nOi_pd = pd.DataFrame()
                coef = []
                for l in range(len(features)):
                    ft = features[l][len(features[l])-3:len(features[l])]
                    if (ft == '5b'):
                        print(ft)
                        coef.append(features[l])
                
                print('Number of 15b Features: ' + str(len(coef)))   

            elif reg_method[m] == '10bc':
                nOi_pd = pd.DataFrame()
                coef = []
                for l in range(len(features)):
                    ft = features[l][len(features[l])-3:len(features[l])]
                    if ((ft == '0b') or (ft == '0c')):
                        print(ft)
                        coef.append(features[l])
                
                print('Number of 10b/c Features: ' + str(len(coef)))   

            elif reg_method[m] == '15bc':
                nOi_pd = pd.DataFrame()
                coef = []
                for l in range(len(features)):
                    ft = features[l][len(features[l])-3:len(features[l])]
                    if ((ft == '5b') or (ft == '5c')):
                        print(ft)
                        coef.append(features[l])
                
                print('Number of 15b/c Features: ' + str(len(coef)))
         
            elif reg_method[m] == '10ib':
                nOi_pd = pd.DataFrame()
                coef = []
                for l in range(len(features)):
                    ft = features[l][len(features[l])-2:len(features[l])]
                    if ((ft == '0b') or (ft == '0i') or (ft == 'rm')):
                        print(ft)
                        coef.append(features[l])
            
                print('Number of 10b and 0i Features: ' + str(len(coef)))
                        
            elif reg_method[m] == '15ib':
                nOi_pd = pd.DataFrame()
                coef = []
                for l in range(len(features)):
                    ft = features[l][len(features[l])-2:len(features[l])]
                    if ((ft == '5b') or (ft == '0i') or (ft == 'rm')):
                        print(ft)
                        coef.append(features[l])
            
                print('Number of 15b and 0i Features: ' + str(len(coef)))
            
            elif reg_method[m] == 'LungRads': 
                nOi_pd = pd.DataFrame()
                coef = []
                for l in range(len(features)):
                    ft = features[l][:]
                    if (ft == 'maximum3ddiameter0i'):
                        print(ft)
                        coef.append(features[l])

            save_dataset(filename = files[n] + '.csv',
                        features = coef,
                        method = reg_method[m],
                        clean=False,
                        normalize=True,
                        resample=2, # (2) to downsample the negative cases
                        seeder = seed
                        )
            
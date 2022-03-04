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

# Libraries
# ---------------------------------------------------------------------------- #
from sklearn.metrics import roc_curve, confusion_matrix, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVC


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import utils
import os
# ---------------------------------------------------------------------------- #

def plot_importance(coefs, std, names, title = None, savepath = None, ext= ".png"):
    plt.title(title)
    plt.xlabel("Importance for Malignancy")
    plt.barh(range(len(coefs)), coefs, xerr = std, align='center')
    plt.yticks(range(len(coefs)), names)
    
    plt.savefig(savepath + ext, bbox_inches='tight')
    plt.close()

def f_importances(feat_imp, dataset= None, method= None):
    
    dataset = dataset.split('_')[1]
    feat_imp = feat_imp.sort_values(by=['mean'])

    # Plot All Features
    title = method.upper() + "Feature Importance"
    result_dir = os.path.split(os.getcwd())[0] + "/results/"
    savepath = result_dir + dataset + "/" + method.upper() + "RankedFeatures"

    plot_importance(feat_imp['mean'],
                    feat_imp['std'],
                    feat_imp['Feature Name'],
                    title= title,
                    savepath= savepath)

    feat_imp.to_csv(savepath + ".csv")   

    # Plot Bottom Featuers
    title = method.upper() + "Bottom 10 Features"
    savepath = result_dir + dataset + "/" + method.upper() + "Bot10Features"
    
    plot_importance(feat_imp['mean'].head(10),                 # Selecting Bottom 10 Features
                    feat_imp['std'].head(10),
                    feat_imp['Feature Name'].head(10),         # Selecting Bottom 10 Feature Names
                    title= title,
                    savepath= savepath)

    # Plot Top Features        
    title = method.upper() + "Top 10 Features"
    savepath = result_dir + dataset + "/" + method.upper() + "Top10Features"
    
    plot_importance(feat_imp['mean'].tail(10),                # Selecting Top 10 Features
                    feat_imp['std'].tail(10),
                    feat_imp['Feature Name'].tail(10),        # Selecting Top 10 Feature Names
                    title= title,
                    savepath= savepath)

def find_best_config(X, y, clf, folds):
    '''
        Run an exhaustive search over the 
        specified parameter values to find the best configuration 
    '''
    if isinstance(clf, RandomForestClassifier):
        param_grid = { 
            'n_estimators': [100, 300, 500],
            'max_features': ['auto', 'sqrt'],
            'criterion'   : ['gini', 'entropy'],
            'class_weight': ['balanced', 'balanced_subsample'],
        }
    elif isinstance(clf, SVC):
        c = [0.1, 1, 10, 50, 75, 100]
        param_grid = {
                'kernel':['linear'],            #RBF would likely produce better results but cannot get Feat importance
                'class_weight': ['balanced'], 
                'degree':[2,3,4,5],
                'gamma':[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                'C':c
            }
    else:
        print('Error: unknown classifier')

    # Find the best parameters with GridSearchCV 
    # for (parallel 5 folds cross-validation)
    optimizer = RandomizedSearchCV(
        clf, param_grid, cv=folds, 
        scoring='roc_auc', verbose=0, n_jobs=-1
    ).fit(X, y)

    df = pd.concat([pd.DataFrame(optimizer.cv_results_["params"]),pd.DataFrame(optimizer.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
    df = df.sort_values(by=['Accuracy'])
    classifier_params= df.head(1)


    return classifier_params

def eval(
    X_train, X_test,     # Defining input Training and Testing Data
    y_train, y_test,     # Defining output class training and testing
    clf,                 # Defining Classifier to use
    feat_names = None,   # Defining Feature Name Variable 
    rep_index = None,    # Defining Rep Index
    seed = 2018,         # Defining Random Seed
    ):
    '''
        Run classifier and claculate the accuracy of the model based 
        on True-Positive and False-Positive predictions

        clf -- 
        -   'rf':  RandomForestClassifier,
        -   'svm': SVMClassifier,
        -   'lasso': LassoRegression
    '''

    if clf == 'rf':
        clf = RandomForestClassifier(random_state=seed)
        clf_params = find_best_config(X_train, y_train, clf, 5)
        
        classifier = RandomForestClassifier(n_estimators= clf_params['n_estimators'].values[0],
                                            max_features=clf_params['max_features'].values[0],
                                            criterion=clf_params['criterion'].values[0],
                                            class_weight=clf_params['class_weight'].values[0])
        
        classifier.fit(X_train, y_train)
        importances = classifier.feature_importances_
        importances = pd.DataFrame(data=importances, columns=[str(rep_index)])

    elif clf == 'svm':
        clf = SVC(random_state=seed, probability=True)
        clf_params = find_best_config(X_train, y_train, clf, 5)
        
        classifier = SVC(kernel= clf_params['kernel'].values[0],
                        degree= clf_params['degree'].values[0],
                        gamma= clf_params['gamma'].values[0],
                        class_weight=clf_params['class_weight'].values[0],
                        C=clf_params['C'].values[0],
                        probability=True)
        
        classifier.fit(X_train, y_train)
        importances = classifier.coef_[0]            
        importances = pd.DataFrame(data=importances, columns=[str(rep_index)])
    
    elif clf == 'Lasso':
        fpr, tpr, tpi, tni, importances = LassoReg(X_train, X_test, y_train, y_test, seed)
        importances = pd.DataFrame(data=importances, columns=[str(rep_index)])
        
        return (fpr, tpr, tpi, tni, importances)
            
    probs = classifier.predict_proba(X_test)   
    y_pred = np.argmax(probs, axis = 1)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpi = tp / (tp + fn)
    tni = tn / (tn + fp)

    print("\nTPI %3f, TNI %3f" %(tpi, tni))
    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
    
    return (fpr, tpr, tpi, tni, importances)

def LassoReg(
        X_train, X_test,    # variables for training and testing
        y_train, y_test,    # Classification for train and test
        seed,               # Random Seed
        ):

    alphas = np.arange(0, 0.1, 0.00001)     #
    avgMSE = []                             #

    for l in range(len(alphas)):
            
        meanSq = np.zeros((len(alphas)))                            #

        lassoReg = Lasso(alpha=alphas[l],
                         normalize=True,
                         max_iter = 100000,
                         tol = 0.0001)

        lassoReg.fit(X_train,y_train)                               #
        train_score = lassoReg.score(X_train, y_train)              #
        test_score = lassoReg.score(X_test,y_test)                  #
        coeff_used = np.sum(lassoReg.coef_!=0)                      #
        y_pred = lassoReg.predict(X_test)                           #
        meanSq[l] = (mean_squared_error(y_test, y_pred))            #
        
        avgMSE.append(np.mean(meanSq[l]))               #
    
    opt_alpha = alphas[np.argmin(avgMSE)]               #

    lassoReg = Lasso(alpha=opt_alpha,
                    normalize=True,
                    max_iter = 100000,
                    tol = 0.0001)

    lassoReg.fit(X_train,y_train)                       #
    train_score = lassoReg.score(X_train, y_train)      #
    test_score = lassoReg.score(X_test,y_test)          #
    
    importance = lassoReg.coef_

    y_pred = lassoReg.predict(X_test)                   #
    fpr, tpr, _ = roc_curve(y_test, y_pred)             #
    
    for i in range(len(y_pred)):
        if (y_pred[i] < 0.5):
            y_pred[i] = 0
        else:
            y_pred[i] = 1

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()   #
    tpi = tp / (tp + fn)                                        #
    tni = tn / (tn + fp)                                        #

    return (fpr, tpr, tpi, tni, importance)



def plot_confmtx(matrix, classes, itr):
    '''
        Plot a confusion matrix for each fold
    '''        
    # plot confusion matrix
    plt.figure(itr+3, figsize=(10,8))
    plt.imshow(matrix, interpolation='nearest', cmap= plt.cm.GnBu)
    
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(
            j, i, format(matrix[i, j], 'd'),
            horizontalalignment="center",
            fontsize=14, color="white" if matrix[i, j] > matrix.max()/2 else "black"
        )
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.title('Fold %d' % (itr+1), fontsize=20)
    plt.tight_layout()

def LungRADS_rules(X_test,y_test):
    """

    """
    num_correct = 0
    y_pred = np.zeros((len(y_test),1))
    
    for i in range(len(X_test)):

        if (X_test[i]<6):
            y_pred[i][0] = 0

        elif(X_test[i] >= 6 and X_test[i] < 8):
            y_pred[i][0] = 0

        elif (X_test[i] >= 8 and X_test[i] < 15):
            y_pred[i][0] = 1

        elif (X_test[i] >= 15):
            y_pred[i][0] = 1

        if y_test[i] == y_pred[i][0]:
            num_correct += 1

    acc = num_correct/len(y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpi = tp / (tp + fn)
    tni = tn / (tn + fp)

    print("Accuracy: %f"%(acc))
    plot_confmtx(confusion_matrix(y_test,y_pred), {'Benign', 'Malignant'}, itr= 10)
    print("TPI %3f, TNI %3f" %(tpi, tni))
    return tpi, tni, acc

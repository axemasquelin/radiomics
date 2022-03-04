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
# libraries and dependencies
# ---------------------------------------------------------------------------- #
from scipy import stats
# from tables import *

import matplotlib.pyplot as plt
import matplotlib.axes as ax
# import seaborn as sns
import pandas as pd
import numpy as np
import glob, os, csv, re
# ---------------------------------------------------------------------------- #\

def combine_dataframe(df1, df2, columns = None, check_pid = False):
    """
    Definition:
    Input:
    Output:
    """
    # if check_pid:
    #     df = df1.merge(df2, on=['pid'])

    df = pd.DataFrame.merge(df1, df2, on='pid')

    return df


def data_load(filename, df_pid= None, var_names = None,
              clean = True, get_pid = False, check_pid = False
              ):
    """
    Definition:
    Input:
    Output:
    """
    vars = []
    df = pd.read_csv(filename, low_memory = False)

    if get_pid == True:  # Normalize Check
        df = df[['ca', 'pid']]

    if check_pid:
        df = df.merge(df_pid, on=['pid'])

        for key, val in var_names.items():
            if var_names[key] in df.columns:
                vars.append(var_names[key])

        df = df[vars]
        if clean:
            # Drop out the columns that have over 90% missing data points.
            df = df.dropna(axis=0)

    return df

def df_stats(df, demographic_table = None, columns = None):
    '''
    Definition:
    Input:
    Output:
    '''
    df1, df2 = [x for _, x in df.groupby(df['ca'] < 1)]

    for key, val in columns.items():
        if ((columns[key] in df.columns) and (key != 'gender') and (key != 'pid')):

            column_Malgn_mean = df1[columns[key]].mean()
            column_Malgn_std = df1[columns[key]].std()
            column_Malgn_var = df1[columns[key]].var()
            print("Malignant - %s: %.3f , +/- %.3f" % (key, column_Malgn_mean, column_Malgn_std))

            column_benign_mean = df2[columns[key]].mean()
            column_benign_std = df2[columns[key]].std()
            column_benign_var = df2[columns[key]].var()
            print("Benign - %s: %.3f , +/- %.3f" % (key, column_benign_mean, column_benign_std))

            if (column_Malgn_var/column_benign_var <= 4):
                eq_var = True
            else:
                eq_var = False

            tStat, pValue = stats.ttest_ind(df1[columns[key]], df2[columns[key]], axis = 0, equal_var= eq_var)
            print("P-value:", pValue)

        elif ((columns[key] in df.columns) and (key == 'gender') and (key != 'pid')):
            malign_genders = df1[columns['gender']].value_counts()
            benign_genders = df2[columns['gender']].value_counts()

def create_demographic_table(df_ca_pids, df_prsn, df_sctabn, df_sctimage, lookup):

    df = df.merge(df_pid, on=['pid'])

    print('\n Get Demographic Data')  # Demographic Data
    df_prsn = combine_dataframe(df_prsn, df_ca_pids, columns=['ca'], check_pid= True)
    df_stats(df_prsn, columns=lookup)

    print('\n Get Nodule Size')
    df_sctabn = df_sctabn.loc[df_sctabn.groupby('pid').SCT_LONG_DIA.idxmax()]
    df_sctabn = combine_dataframe(df_sctabn, df_ca_pids, columns=['ca'], check_pid= True)
    df_stats(df_sctabn, columns=lookup)

    print('\n Get LDCT Parameters')
    df_sctimage = df_sctimage.loc[df_sctimage.groupby('pid').scr_group.idxmax()]
    df_sctimage = combine_dataframe(df_sctimage, df_ca_pids, columns=['ca'], check_pid= True)
    df_stats(df_sctimage, columns=lookup)

def relable_study_yr(df_sctabn):

    t_yr = []
    dx_yr = []
    for i in range(len(df_sctabn['STUDY_YR'])):
        if df_sctabn['STUDY_YR'][i] == 0:
            t_yr.append('T0')
        elif df_sctabn['STUDY_YR'][i] == 1:
            t_yr.append('T1')
        elif df_sctabn['STUDY_YR'][i] == 2:
            t_yr.append('T2')

        if df_sctabn['scr_group'][i] == 0:
            dx_yr.append('T0')
        if df_sctabn['scr_group'][i] == 1:
            dx_yr.append('T1')
        if df_sctabn['scr_group'][i] == 2:
            dx_yr.append('T2')
        if df_sctabn['scr_group'][i] == 3:
            dx_yr.append('T0')
        if df_sctabn['scr_group'][i] == 4:
            dx_yr.append('T1')
        if df_sctabn['scr_group'][i] == 5:
            dx_yr.append('T2')
        if df_sctabn['scr_group'][i] == 6:
            dx_yr.append('NA')
        if df_sctabn['scr_group'][i] == 7:
            dx_yr.append('NA')

    df_sctabn['t_yr'] = t_yr
    df_sctabn['dx_yr'] = dx_yr

    return df_sctabn

def create_exclusion_flowchart(df_ca_pids, df_prsn, df_sctabn, df_sctimage):
    exclusion_criteria = {
        'rndgroup': 1,
        'SCT_LONG_DIA': 20,
        'solid_vs_grndglass': 'Solid',
    }
    print('NLST # Participants: %i' % (len(df_prsn)))
    df_group_exclusion = df_prsn.loc[df_prsn['rndgroup'] == 2]
    df_prsn = df_prsn.loc[df_prsn['rndgroup'] == 1]
    print('LDCT Group #: %i vs. Rad Group #: %i' %(len(df_prsn), len(df_group_exclusion)))

    uniqueId_sctabn = df_sctabn["pid"].unique()
    print("Number of Unique PID in LDCT Branch: %i " %len(uniqueId_sctabn))
    print(len(df_sctabn))
    df_sctabn = df_sctabn.merge(df_ca_pids, on=['pid'])
    print(len(df_sctabn))
    df_sctabn = df_sctabn.dropna(axis = 0, subset = ['SCT_LONG_DIA', 'SCT_AB_DESC', 'scr_group', 'STUDY_YR'])
    df_sctabn = df_sctabn.dropna(axis = 0, subset = ['SCT_LONG_DIA', 'SCT_AB_DESC', 'scr_group', 'STUDY_YR'])
    df_sctabn = df_sctabn.loc[df_sctabn['SCT_PRE_ATT'] == 1]
    df_sctabn = df_sctabn.reset_index()
    df_sctabn = relable_study_yr(df_sctabn)
    df_sctabn = df_sctabn.loc[df_sctabn['t_yr'] == df_sctabn['dx_yr']]
    print("Lenght of Dataset: %i" %len(df_sctabn))

    df_sctabn = df_sctabn.loc[df_sctabn.groupby('pid').SCT_LONG_DIA.idxmax()]

    df_sctabn = df_sctabn.merge(df_ca_pids, on=['pid'])

    print(df_sctabn[['pid','SCT_LONG_DIA', 'SCT_AB_DESC']])

    uniqueId_sctabn = df_sctabn["pid"].unique()
    print("Number of Unique PID in LDCT Branch: %i " % len(uniqueId_sctabn))

    df_group_exclusion = df_sctabn.loc[df_sctabn['SCT_LONG_DIA'] > 20]
    df_sctabn = df_sctabn.loc[df_sctabn['SCT_LONG_DIA'] <= 20]
    df_sctabn = df_sctabn.loc[df_sctabn['SCT_LONG_DIA'] >= 4]
    print("Nodules 4mm<= x <= 20mm Count: %i, Nodules excluded: %i" %(len(df_sctabn), len(df_group_exclusion)))


    print("Actual Dataset Lenght: %i" %(len(df_ca_pids)))

def abnormalities_check(df_ca_pids, df_sctabn):


    uniqueId_sctabn = df_sctabn["pid"].unique()
    print("Number of Unique PID in LDCT Branch: %i " % len(uniqueId_sctabn))

    df_sctabn = df_sctabn.merge(df_ca_pids, on=['pid'])

    idx_51 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 51)))
    idx_52 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 52)))
    idx_53 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 53)))
    idx_54 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 54)))
    idx_55 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 55)))
    idx_56 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 56)))
    idx_57 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 57)))
    idx_58 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 58)))
    idx_59 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 59)))
    idx_60 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 60)))
    idx_61 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 61)))
    idx_62 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 62)))
    idx_63 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 63)))
    idx_64 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 64)))
    idx_65 = np.count_nonzero(np.where((df_sctabn['SCT_AB_DESC'] == 65)))

    print("idx 51,...,64: %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i"
          %(idx_51, idx_52, idx_53, idx_54,idx_55, idx_56, idx_57, idx_58, idx_59,
            idx_60, idx_61, idx_62, idx_63, idx_64, idx_65))

def main():
    """
    Definition:
    Input:
    Output:
    """
    variables = {
            'age': 'age',                           # Patient Age
            'pid': 'pid',                           # Patient Identifier
            'pack_years': 'pkyr',                   # Smoking Pack Years
            'gender': 'gender',                     # Gender
            'ethnicity': 'ETHNIC',                  # Ethnicity
            'nodule_size': 'SCT_LONG_DIA',          # Nodule Size
            'abnormalities': 'SCT_AB_DESC',
            'scr_group': 'scr_group',               # Screening Group
            'study_yr': 'study_yr',                 # Study Year of Screening Exam
            'Kilo_VP': 'kvp',                       # Kilo Voltage
            'slice_thickness': 'reconthickness',    # Reconstructed Slice Thickness
            'Tube Current': 'mas',                  # Tube Current (mA)
            'rndgroup': 'rndgroup',                  # Study Arm
            'selected': 'selected',
            'attenuation': 'SCT_PRE_ATT'
            }
    lookup = {
            'age': 'age',                           # Patient Age
            'pack_years': 'pkyr',                   # Smoking Pack Years
            'nodule_size': 'SCT_LONG_DIA',          # Nodule Size
            'Kilo_VP': 'kvp',                       # Kilo Voltage
            'slice_thickness': 'reconthickness',    # Reconstructed Slice Thickness
            'Tube Current': 'mas'                   # Tube Current (mA)
    }

    # Directory Initialization
    cwd = os.getcwd()                                       # Initialize Current Working Directory
    folder_cohort = os.path.split(cwd)[0] + "\Cohort Data/" # NLST Dataset Information Folder
    folder_dataset = cwd + '/dataset/'

    # Get Radiomic Data File
    rad_file = folder_dataset + 'data_Original.csv'         # Radiomic Dataset File (Only need PIDs)
    files_cohort = glob.glob(folder_cohort + "/*.csv")      # Cohort Data Files


    # demographic_table = np.zeros((4,2))
    df_ca_pids = data_load(rad_file, get_pid= True)                      # DF with PIDs and Classification
    df_prsn = data_load(files_cohort[0], df_ca_pids, variables)          # DF with Patient Demographics
    df_sctabn = data_load(files_cohort[1], df_ca_pids, variables)        # DF with CT Abnormalities
    df_sctimage = data_load(files_cohort[2], df_ca_pids, variables)      # DF with CT Image Information

    create_exclusion_flowchart(df_ca_pids, df_prsn, df_sctabn, df_sctimage)
    create_demographic_table(df_ca_pids, df_prsn, df_sctabn, df_sctimage, lookup)


if __name__ == "__main__":
    """
    Definition:        
    Input:
    Output:
    """
    main()

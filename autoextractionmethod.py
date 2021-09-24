# -----------------------------------------------------------
# Automatic Extraction Method
#
# (C) 2021 Camila Castro Moreno, Sao Jose dos Campos, Brazil
# Released under MIT License
# email camilacastromoreno1994@gmail.com
# -----------------------------------------------------------

import operator
import pandas as pd
from collections import defaultdict
from IPython.display import display_html
from itertools import chain,cycle
from sklearn.metrics import f1_score
import numpy as np

#########################################################################################################################################################################################################


def auto_extraction(df_data, df_metadata, df_performance, percent_drop, percent_merge):
    
    """
    This function is used to extract intervals of easy and hard instances for each dataset using each performance column
    
    :param df_data: original dataset
    :type df_data: pandas.core.frame.DataFrame
 
    :param df_metadata: metafeature dataset
    :type df_metadata: pandas.core.frame.DataFrame

    :param df_performance: algorithm bin and beta easy values 
    :type df_performance: pandas.core.frame.DataFrame
    
    :param percent_drop: percentage to calculate size of intervals that should be dropped 
    :type percent_drop: pandas.core.series.Series or numpy.float64
    
    :param percent_merge: percentage to calculate size of gaps between intervals that should be merged 
    :type percent_merge: pandas.core.series.Series or numpy.float64

    :return dict_E, dict_H: dictionaries with easy and hard intervals for each combination of performance measure and metafeature
    :rtype: dict
    """   
    
      
    
    # Let's define the easy and hard interval dictionary
    dict_E = {}
    dict_H = {}

    # We'll be evaluating the domains of competence of each performance measure within the dataset
    # Let's get the performance names for the current dataset
    performance_names = list(df_performance.columns)
    
    # Let's go through each class (performance)
    for performance in performance_names:

        # Let's create an empty dictionary for this performance measure
        dict_E[performance] = {}
        dict_H[performance] = {}

        # Let's get the meta feature names for the current dataset
        meta_feature_names = list(df_metadata.columns)

        # let's go through each column of the meta features (MFj)
        for metafeature in meta_feature_names:
            
            # Let's create an empty dictionary for this metafeature 
            # in this algorithm
            dict_E[performance][metafeature] = {}
            dict_H[performance][metafeature] = {}
            
            # Get interval values and indexes
            if isinstance(percent_drop, pd.Series):
                E_int_ind, E_int_val, H_int_ind, H_int_val = get_intervals(df_data, 
                                                                            df_metadata[metafeature], 
                                                                            df_performance[performance], 
                                                                            percent_drop[performance], 
                                                                            percent_merge[performance])
                    
            else: 
                E_int_ind, E_int_val, H_int_ind, H_int_val = get_intervals(df_data, 
                                                                           df_metadata[metafeature], 
                                                                           df_performance[performance], 
                                                                           percent_drop, 
                                                                           percent_merge)
            
            # Add E_aux and H_aux to dictionaries according to the current 
            # algorithm, metafeature and interval        
            dict_E[performance][metafeature]['interval_ind'] = E_int_ind
            dict_E[performance][metafeature]['interval_val'] = E_int_val
            dict_H[performance][metafeature]['interval_ind'] = H_int_ind
            dict_H[performance][metafeature]['interval_val'] = H_int_val
    
    
    return dict_E, dict_H

#########################################################################################################################################################################################################

def get_intervals(df_data, df_metafeature, df_performance, percent_drop, percent_merge):
    
    """
    This function is used to get intervals of easy and hard instances

    :param df_data: original dataset
    :type df_data: pandas.core.frame.DataFrame

    :param df_metadata: metafeature dataset
    :type df_metadata: pandas.core.frame.DataFrame

    :param df_performance: algorithm bin and beta easy values 
    :type df_performance: pandas.core.frame.DataFrame

    :param percent_drop: percentage to calculate size of intervals that should be dropped 
    :type percent_drop: numpy.float64

    :param percent_merge: percentage to calculate size of gaps between intervals that should be merged 
    :type percent_merge: numpy.float64

    :return E_int_ind_aux, E_int_val_aux, H_int_ind_aux, H_int_val_aux: intervals and values of easy and hard behavior
    :rtype: list
    """
    
    # let's create an auxiliary empty list of easy and hard interval indexes and values
    E_int_ind = []
    E_int_val = []
    H_int_ind = []
    H_int_val = []

    # add MFj that will be used for sorting to dataset and 
    # algorithm performance from algorithm_bin to define easy/hard elements 
    U = df_data.assign(
        MFj = df_metafeature, 
        Algorithm_Perf = df_performance)

    # sort list U by each meta feature MFj
    UMFj = U.sort_values(by=['MFj'])
    UMFj.reset_index(inplace=True)
    # let's search for easy behavior intervals
    i = 0
    pos = 0
    while i < len(UMFj)-1 and pos != -1:

        # position of the next easy instance
        pos = next_easy_instance(i, UMFj)

        if pos != -1:
            V_ind, V_val, i = extend_easy_interval(pos, UMFj)
            E_int_ind.append(V_ind)
            E_int_val.append(V_val)

    # let's search for hard behavior intervals
    i = 0
    pos = 0
    while i < len(UMFj)-1 and pos != -1:

        # position of the next hard instance
        pos = next_hard_instance(i, UMFj)
        if pos != -1:
            V_ind, V_val, i = extend_hard_interval(pos, UMFj)
            H_int_ind.append(V_ind)
            H_int_val.append(V_val)
    
    
    # Let's merge and drop the intervals if necessary
    count = df_performance.value_counts()
    n_easy = count[1]
    n_hard = count[0]
    
    
    E_int_ind, E_int_val = merge_intervals(E_int_ind, E_int_val, len(df_performance), percent_merge)
    H_int_ind, H_int_val = merge_intervals(H_int_ind, H_int_val, len(df_performance), percent_merge)

    E_int_ind, E_int_val = drop_small_intervals(E_int_ind, E_int_val, n_easy, percent_drop)
    H_int_ind, H_int_val = drop_small_intervals(H_int_ind, H_int_val, n_hard, percent_drop)
    
    # keep easy interval with the largest support
    if E_int_ind:
        E_int_ind_aux = E_int_ind[0]
        E_int_val_aux = E_int_val[0]

        for i in range(len(E_int_ind)):
            if E_int_ind[i][1] - E_int_ind[i][0] > E_int_ind_aux[1] - E_int_ind_aux[0]:
                E_int_ind_aux = E_int_ind[i]
                E_int_val_aux = E_int_val[i]
    else: 
        E_int_ind_aux = []
        E_int_val_aux = []

    # keep hard interval with the largest support
    if H_int_ind:
        H_int_ind_aux = H_int_ind[0]
        H_int_val_aux = H_int_val[0]

        for i in range(len(H_int_ind)):
            if H_int_ind[i][1] - H_int_ind[i][0] > H_int_ind_aux[1] - H_int_ind_aux[0]:
                H_int_ind_aux = H_int_ind[i]
                H_int_val_aux = H_int_val[i]
    else: 
        H_int_ind_aux = []
        H_int_val_aux = []
    
    return E_int_ind_aux, E_int_val_aux, H_int_ind_aux, H_int_val_aux

#########################################################################################################################################################################################################


def next_easy_instance(i, UMFj):
    
    """
    This function is used to determine the next easy instance j (Algorithmic Performance == 1) of a sorted dataset UMFj
        
    :param i: starting index to search for next easy instance
    :type i: int

    :param UMFj: orignal dataset sorted by a metafeature column
    :type UMFj: pandas.core.frame.DataFrame

    :return j or -1: j if the next easy instance is found and -1 if not
    :type j: int
    """                   
                     
                     
    j = i
    while j < len(UMFj):
        if UMFj['Algorithm_Perf'][j] == 1:
            return j;
        j += 1;
    
    return -1;

#########################################################################################################################################################################################################

def next_hard_instance(i, UMFj):
                     
    """
    This function is used to determine the next hard instance j (Algorithmic Performance == 0) of a sorted dataset UMFj
        
    :param i: starting index to search for next hard instance
    :type i: int

    :param UMFj: orignal dataset sorted by a metafeature column
    :type UMFj: pandas.core.frame.DataFrame

    :return j or -1: j if the next hard instance is found and -1 if not
    :type j: int
    """      
                     
    j = i
    while j < len(UMFj):
        if UMFj['Algorithm_Perf'][j] == 0:
            return j;
        j += 1;
    
    return -1;

#########################################################################################################################################################################################################

def extend_easy_interval(pos, UMFj):
    
    """
    This function is used to extend easy intervals by along moving along indexes of UMFj starting from a position pos
        
    :param pos: starting index for extending a easy interval
    :type pos: int

    :param UMFj: orignal dataset sorted by a metafeature column
    :type UMFj: pandas.core.frame.DataFrame

    :return interval: index of first and last elements of the extended interval
    :type interval: list
        
    :return values: value of first and last elements of the extended interval
    :type values: list
        
    :return limit: index of last element in the extended interval
    :type limit: int
    """      
                     
                     
    limit = pos
    end = False
    while end == False and UMFj['Algorithm_Perf'][limit] == 1:
        limit += 1;
        if limit > len(UMFj)-1:
            limit = len(UMFj)-1
            end = True
    
                     
    interval = [pos, limit]
    values = [round(UMFj['MFj'][pos], 6), round(UMFj['MFj'][limit], 6)]
                     
    return interval, values, limit

#########################################################################################################################################################################################################

def extend_hard_interval(pos, UMFj):
                                       
    """
    This function is used to extend hard intervals by along moving along indexes of UMFj starting from a position pos
        
    :param pos: starting index for extending a hard interval
    :type pos: int

    :param UMFj: orignal dataset sorted by a metafeature column
    :type UMFj: pandas.core.frame.DataFrame

    :return interval: index of first and last elements of the extended interval
    :type interval: list
        
    :return values: value of first and last elements of the extended interval
    :type values: list
        
    :return limit: index of last element in the extended interval
    :type limit: int
    """      
                     
    limit = pos
    end = False
    while end == False and UMFj['Algorithm_Perf'][limit] == 0:
        limit += 1;
        if limit > len(UMFj)-1:
            limit = len(UMFj)-1
            end = True
                     
    interval = [pos, limit]
    values = [round(UMFj['MFj'][pos], 6), round(UMFj['MFj'][limit], 6)]
                     
    return interval, values, limit

#########################################################################################################################################################################################################

def merge_intervals(EH_ind, EH_val, n, percent_merge):
                     
    """
    This function is used to merge intervals with a maximum gap of n*percent_merge elements
        
    :param EH_ind: index of first and last elements of the original interval
    :type EH_ind: list

    :param EH_val: value of first and last elements of the original interval
    :type EH_val: list
        
    :param n: length of the dataset
    :type n: int
        
    :param percent_merge: percentage to calculate size of gaps between intervals that should be merged 
    :type percent_merge: numpy.float64

    :return EH_ind: index of first and last elements of the new interval
    :type EH_ind: list
        
    :return EH_val: value of first and last elements of the new interval
    :type EH_val: list
    """      
    
    i = 0
    while i < len(EH_ind)-1:
        
        # Here are the interval indexes and values 
        inter1_ind = EH_ind[i]
        inter1_val = EH_val[i]
        inter2_ind = EH_ind[i+1]
        inter2_val = EH_val[i+1]
        
        # if the gap between two intervals is a maximum of n*percent_merge
        if inter2_ind[0] - inter1_ind[1] <= n*percent_merge:
            
            # the new intervals will be the beginning of the first interval with the ending of the  
            # second interval
            newInter_ind = [inter1_ind[0], inter2_ind[1]]
            newInter_val = [inter1_val[0], inter2_val[1]]
            
            #insert new intervals
            EH_ind.insert(i+2, newInter_ind)
            EH_val.insert(i+2, newInter_val)
            
            # remove two prior intervals
            EH_ind.remove(inter1_ind)
            EH_ind.remove(inter2_ind)
            EH_val.remove(inter1_val)
            EH_val.remove(inter2_val)
            
        else: 
            i+=1
            
    return EH_ind, EH_val;

#########################################################################################################################################################################################################

def drop_small_intervals(EH_ind, EH_val, n, percent_drop):
                     
                              
    """
    This function is used to remove intervals with less than n*percent_drop, where n is number of instances of the class
        
    :param EH_ind: index of first and last elements of the original interval
    :type EH_ind: list

    :param EH_val: value of first and last elements of the original interval
    :type EH_val: list
        
    :param n: length of the dataset
    :type n: int
        
    :param percent_drop: percentage to calculate size of intervals that should be dropped 
    :type percent_drop: numpy.float64

    :return EH_ind: index of first and last elements of the new interval
    :type EH_ind: list
        
    :return EH_val: value of first and last elements of the new interval
    :type EH_val: list
    """      
                     
    i = 0
    while i < len(EH_ind):
        if len(range(EH_ind[i][0], EH_ind[i][1]+1)) <= n*percent_drop:
            EH_ind.remove(EH_ind[i])
            EH_val.remove(EH_val[i])
        else: 
            i+=1
    
    return EH_ind, EH_val;

#########################################################################################################################################################################################################

def simple_rules(EH, n):
    
    """
    This function returns a dictionary of easy/hard simple rules for each performance given the easy/hard intervals EH of a dataset and the length n of the dataset
    
    :param EH: dictionary with easy or hard intervals for each combination of performance measure and metafeature
    :type EH: dict
    
    :param n: length of the dataset
    :type n: int
    
    :return dict_simple_rules: a dictionary of dataframes for each algorithm with interval values, indices and support of a metafeatures easy or hard behavior 
    :rtype dict_simple_rules: dict
    """  

    dict_simple_rules = {}
    
    for performance in EH:
        
        # empty metafeature list (metafeatures can be repeated in this list if there is 
        # a metafeature with more than one interval associated to it)
        metafeature_list = []
        
        # empty interval list for this algorithm
        interval_list = []
        
        # empty interval index list for this algorithm
        interval_ind_list = []
        
        # empty support list (%) which is the percentage of instances that meet the rule
        support_list = []
        
        for metafeature in EH[performance]:
            
            # Get interval values and interval indexes
            int_val = EH[performance][metafeature]['interval_val']
            int_ind = EH[performance][metafeature]['interval_ind']
            
            # verify if the interval is non-empty
            # if not append to each list
            if int_val:
                interval_list.append(int_val)
                metafeature_list.append(metafeature)
                interval_ind_list.append(int_ind)
                support_list.append(round((len(range(int_ind[0], int_ind[1]+1))/n), 2))
                    
                
        dict_simple_rules[performance] = pd.DataFrame(list(zip(metafeature_list, 
                                                               interval_ind_list, 
                                                               interval_list, 
                                                               support_list)), 
                                                      columns =['Metafeature', 
                                                                'Index', 
                                                                'Interval', 
                                                                'Support'])
        
    return dict_simple_rules

#########################################################################################################################################################################################################

def common_rules(dict_easy_rules, dict_hard_rules):
    
    """
    This function returns the rules for features that are present in both easy and hard intervals for each performance measure
    
    :param dict_easy_rules: a dictionary of dataframes for each algorithm with interval values, indices and support of a metafeatures easy behavior 
    :type dict_easy_rules: dict
    
    :param dict_hard_rules: a dictionary of dataframes for each algorithm with interval values, indices and support of a metafeatures hard behavior 
    :type dict_hard_rules: dict
    
    :return dict_easy_rules_common:  dictionary of dataframes for each algorithm with interval values, indices and support of a metafeatures easy behavior that is common to both easy and hard rules
    :rtype dict_easy_rules_common: dict
    
    :return dict_hard_rules_common:  dictionary of dataframes for each algorithm with interval values, indices and support of a metafeatures hard behavior that is common to both easy and hard rules
    :rtype dict_hard_rules_common: dict
    
    
    """
    
    dict_easy_rules_common = {}
    dict_hard_rules_common = {}

    for performance in dict_easy_rules:
        # features in common
        features_easy = list(dict_easy_rules[performance]['Metafeature'])
        features_hard = list(dict_hard_rules[performance]['Metafeature'])

        common_features = [feature for feature in features_easy if feature in features_hard]

        dict_easy_rules_common[performance] = dict_easy_rules[performance][dict_easy_rules[performance]['Metafeature'].isin(common_features)]
        dict_hard_rules_common[performance] = dict_hard_rules[performance][dict_hard_rules[performance]['Metafeature'].isin(common_features)]
        
    return dict_easy_rules_common, dict_hard_rules_common

#########################################################################################################################################################################################################

def predict(df_performance_test, df_metadata_test, dict_easy_rules, dict_hard_rules):
    
    """
    This function returns the predictions of performance of the test dataset based on the rules generated from the train dataset
    
    :param df_performance_test: algorithm bin and beta easy test dataset 
    :type df_performance_test: pandas.core.frame.DataFrame
    
    :param df_metadata_test: metafeature test dataset
    :type df_metadata_test: pandas.core.frame.DataFrame

    :param dict_easy_rules: dictionary of dataframes for each algorithm with interval values, indices and support of a metafeatures easy behavior
    :type dict_easy_rules: dict
    
    :param dict_hard_rules: dictionary of dataframes for each algorithm with interval values, indices and support of a metafeatures hard behavior
    :type dict_hard_rules: dict
    
    :return dict_pred: dictionary of dataframes that reports if an instance if easy according to the easy rules and hard according to the hard rules
    :rtype dict_pred: dict
    """   

    # Let's define the predicted results dictionary
    dict_pred = {}

    # Let's get the performance names for the current dataset
    performance_names = list(df_performance_test.columns)

    # Let's go through each class (performance)
    for performance in performance_names:

        # Let's create a dataframe with zeros for this performance in this dataset
        dict_pred[performance] = pd.DataFrame(0, index=range(len(df_metadata_test)), columns=["PRD", "NRD"])

        df_easy = dict_easy_rules[performance]
        df_hard = dict_hard_rules[performance]

        # go through each instance
        for i in range(len(df_metadata_test)):

            # Join in a dataframe the values of a given instance of each metafeature to the interval
            df_instance_meta = pd.DataFrame(df_metadata_test.iloc[i,:])
            df_instance_meta.reset_index(level=0, inplace=True)
            df_instance_meta.columns = ['Metafeature', 'Values']

            df_int_instance_easy = df_easy.merge(df_instance_meta, on = "Metafeature")
            df_int_instance_hard = df_hard.merge(df_instance_meta, on = "Metafeature")

            # go through each metafeature to check if in interval, if yes break

            for j in range(len(df_int_instance_easy)):
                if df_int_instance_easy['Interval'][j][0] <= df_int_instance_easy['Values'][j] <= df_int_instance_easy['Interval'][j][1]:
                    dict_pred[performance]["PRD"][i] = 1
                    break
                else: 
                    continue

            for j in range(len(df_int_instance_hard)):
                if df_int_instance_hard['Interval'][j][0] <= df_int_instance_hard['Values'][j] <= df_int_instance_hard['Interval'][j][1]:
                    dict_pred[performance]["NRD"][i] = 1
                    break
                else: 
                    continue
    
    return dict_pred

#########################################################################################################################################################################################################


def tune_hyper_params(merge_space, drop_space, df_train, df_metadata_train, df_performance_train, df_metadata_val, df_performance_val, direc):
    
    """
    This function saves the percent_merge and percent_drop value combination for each performance measure with greatest f1 score by performing a grid search
    
    :param merge_space: search space for percent_merge hyperparameter
    :type merge_space: numpy.ndarray
    
    :param drop_space: search space for percent_drop hyperparameter
    :type drop_space: numpy.ndarray
    
    :param df_train: train dataset
    :type df_train: pandas.core.frame.DataFrame
 
    :param df_metadata_train: metafeature train dataset
    :type df_metadata_train: pandas.core.frame.DataFrame

    :param df_performance_train: algorithm bin and beta easy train dataset 
    :type df_performance_train: pandas.core.frame.DataFrame
    
    :param df_metadata_train: metafeature train dataset
    :type df_metadata_train: pandas.core.frame.DataFrame

    :param df_performance_val: algorithm bin and beta easy validation dataset 
    :type df_performance_val: pandas.core.frame.DataFrame
    
    :param direc: directory where the best parameters will be saved 
    :type direc: str
    
    :return df_f1: f1 scores of the grid serach for each algorithm
    :rtype: pandas.core.frame.DataFrame
    """   
    
    list_merge_value = []
    list_drop_value = []
    list_f1 = []

    for percent_merge in merge_space:

        for percent_drop in drop_space:

            print(f'percent merge: {percent_merge:.3f}, percent drop: {percent_drop:.2f}')

            # extract the rule intervals using the train data
            dict_E, dict_H, = auto_extraction(df_train, 
                                              df_metadata_train, 
                                              df_performance_train, 
                                              percent_drop, 
                                              percent_merge)

            # easy behavior rules
            dict_easy_rules = simple_rules(dict_E, len(df_train))

            # hard behavior rules
            dict_hard_rules = simple_rules(dict_H, len(df_train))

            # Let's keep the features that are present in both easy and hard intervals for a given performance measure
            dict_easy_rules_common, dict_hard_rules_common = common_rules(dict_easy_rules, dict_hard_rules)

            # Let's predict the performance of the validation dataset
            dict_pred = predict(df_performance_val, 
                                df_metadata_val, 
                                dict_easy_rules_common, 
                                dict_hard_rules_common)

            # Let's evaluate the performance ruleset
            # Let's get the performance names for the current dataset
            performance_names = list(df_performance_val.columns)

            # Here's a temporary list to save the f1 scores of each performance metric
            list_f1_temp = []

            for performance in performance_names:
                dict_pred[performance].PRD = dict_pred[performance].PRD == 1
                dict_pred[performance].NRD = dict_pred[performance].NRD == 1

                dict_pred[performance]['PRD_not_NRD'] = dict_pred[performance].PRD & ~dict_pred[performance].NRD
                dict_pred[performance]['NRD_not_PRD'] = ~dict_pred[performance].PRD & dict_pred[performance].NRD

                # Which rules are used to classify easy and hard instances
                mask1 = dict_pred[performance].PRD_not_NRD == True
                mask2 = dict_pred[performance].NRD_not_PRD == True

                dict_pred[performance]['Final'] = [-1] * len(df_metadata_val)

                dict_pred[performance].loc[mask2, ('Final')] = 0
                dict_pred[performance].loc[mask1, ('Final')] = 1

                # f1 score
                f1 = f1_score(df_performance_val[performance], 
                               dict_pred[performance].Final, 
                               average='weighted')

                # Let's append to the temporary list
                list_f1_temp.append(acc)

            # add the f1 scores to the final f1 score list
            list_f1.append(list_f1_temp)

            # add percent_merge to list
            list_merge_value.append(percent_merge)

            # add percent_drop to list
            list_drop_value.append(percent_drop)
    
    
    # Get the best f1 scsores and save their respective merge and drop percentages
    df_f1 = pd.DataFrame(list_f1, columns=performance_names)

    list_best_merge = []
    list_best_drop = []

    for performance in performance_names:
        best_index = df_f1.index[df_f1[performance] == max(df_f1[performance])].tolist()[0]
        list_best_merge.append(list_merge_value[best_index])
        list_best_drop.append(list_drop_value[best_index])
    
    df_best = pd.DataFrame([list_best_merge, list_best_drop], 
                           columns=performance_names, 
                           index=['merge', 'drop'])
    
    
    df_best.to_csv(f'{direc}/best_hyperparameters.csv', index=False)
    
    return df_f1
#########################################################################################################################################################################################################


# This function is used to lay tables side-by-side in Jupyter-Notebook
def display_side_by_side(*args,titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)
    

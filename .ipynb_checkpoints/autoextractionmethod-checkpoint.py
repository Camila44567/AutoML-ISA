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
from sklearn.metrics import accuracy_score
import numpy as np

#########################################################################################################################################################################################################


def auto_extraction(df_data, df_metadata, df_performance, percent_drop, percent_merge):
    
    """
    This function is used to extract intervals of Good and Bad instances for each dataset using each performance column
    
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

    :return dict_G, dict_B: dictionaries with good and bad intervals for each combination of performance measure and metafeature
    :rtype: dict
    """   
    
      
    
    # Let's define the good and bad interval dictionary
    dict_G = {}
    dict_B = {}

    # We'll be evaluating the domains of competence of each performance measure within the dataset
    # Let's get the performance names for the current dataset
    performance_names = list(df_performance.columns)
    
    # Let's go through each class (performance)
    for performance in performance_names:

        # Let's create an empty dictionary for this performance measure
        dict_G[performance] = {}
        dict_B[performance] = {}

        # Let's get the meta feature names for the current dataset
        meta_feature_names = list(df_metadata.columns)

        # let's go through each column of the meta features (MFj)
        for metafeature in meta_feature_names:
            
            # Let's create an empty dictionary for this metafeature 
            # in this algorithm
            dict_G[performance][metafeature] = {}
            dict_B[performance][metafeature] = {}
            
            # Get interval values and indexes
            if isinstance(percent_drop, pd.Series):
                G_int_ind, G_int_val, B_int_ind, B_int_val = get_intervals(df_data, 
                                                                            df_metadata[metafeature], 
                                                                            df_performance[performance], 
                                                                            percent_drop[performance], 
                                                                            percent_merge[performance])
                    
            else: 
                G_int_ind, G_int_val, B_int_ind, B_int_val = get_intervals(df_data, 
                                                                           df_metadata[metafeature], 
                                                                           df_performance[performance], 
                                                                           percent_drop, 
                                                                           percent_merge)
            
            # Add G_aux and B_aux to dictionaries according to the current 
            # algorithm, metafeature and interval        
            dict_G[performance][metafeature]['interval_ind'] = G_int_ind
            dict_G[performance][metafeature]['interval_val'] = G_int_val
            dict_B[performance][metafeature]['interval_ind'] = B_int_ind
            dict_B[performance][metafeature]['interval_val'] = B_int_val
    
    
    return dict_G, dict_B

#########################################################################################################################################################################################################

def get_intervals(df_data, df_metafeature, df_performance, percent_drop, percent_merge):
    
        """
        This function is used to get intervals of Good and Bad instances

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

        :return G_int_ind_aux, G_int_val_aux, B_int_ind_aux, B_int_val_aux: intervals and values of good and bad behavior
        :rtype: list
        """   

    # let's create an auxiliary empty list of good and bad interval indexes and values 
    G_int_ind = []
    G_int_val = []
    B_int_ind = []
    B_int_val = []

    # add MFj that will be used for sorting to dataset and 
    # algorithm performance from algorithm_bin to define good/bad elements 
    U = df_data.assign(
        MFj = df_metafeature, 
        Algorithm_Perf = df_performance)

    # sort list U by each meta feature MFj
    UMFj = U.sort_values(by=['MFj'])
    UMFj.reset_index(inplace=True
    # let's search for good behavior intervals
    i = 0
    pos = 0
    while i < len(UMFj)-1 and pos != -1:

        # position of the next good instance
        pos = next_good_instance(i, UMFj)

        if pos != -1:
            V_ind, V_val, i = extend_good_interval(pos, UMFj)
            G_int_ind.append(V_ind)
            G_int_val.append(V_val)

    # let's search for bad behavior intervals
    i = 0
    pos = 0
    while i < len(UMFj)-1 and pos != -1:

        # position of the next bad instance
        pos = next_bad_instance(i, UMFj)
        if pos != -1:
            V_ind, V_val, i = extend_bad_interval(pos, UMFj)
            B_int_ind.append(V_ind)
            B_int_val.append(V_val)
    
    
    # Let's merge and drop the intervals if necessary
    count = df_performance.value_counts()
    n_good = count[1]
    n_bad = count[0]
    
    
    G_int_ind, G_int_val = merge_intervals(G_int_ind, G_int_val, len(df_performance), percent_merge)
    B_int_ind, B_int_val = merge_intervals(B_int_ind, B_int_val, len(df_performance), percent_merge)

    G_int_ind, G_int_val = drop_small_intervals(G_int_ind, G_int_val, n_good, percent_drop)
    B_int_ind, B_int_val = drop_small_intervals(B_int_ind, B_int_val, n_bad, percent_drop)
    
    # keep Good interval with the largest support
    if G_int_ind:
        G_int_ind_aux = G_int_ind[0]
        G_int_val_aux = G_int_val[0]

        for i in range(len(G_int_ind)):
            if G_int_ind[i][1] - G_int_ind[i][0] > G_int_ind_aux[1] - G_int_ind_aux[0]:
                G_int_ind_aux = G_int_ind[i]
                G_int_val_aux = G_int_val[i]
    else: 
        G_int_ind_aux = []
        G_int_val_aux = []

    # keep Bad interval with the largest support
    if B_int_ind:
        B_int_ind_aux = B_int_ind[0]
        B_int_val_aux = B_int_val[0]

        for i in range(len(B_int_ind)):
            if B_int_ind[i][1] - B_int_ind[i][0] > B_int_ind_aux[1] - B_int_ind_aux[0]:
                B_int_ind_aux = B_int_ind[i]
                B_int_val_aux = B_int_val[i]
    else: 
        B_int_ind_aux = []
        B_int_val_aux = []
    
    return G_int_ind_aux, G_int_val_aux, B_int_ind_aux, B_int_val_aux

#########################################################################################################################################################################################################


def next_good_instance(i, UMFj):
    
        """
        This function is used to determine the next good instance j (Algorithmic Performance == 1) of a sorted dataset UMFj
        
        :param i: starting index to search for next good instance
        :type i: int

        :param UMFj: orignal dataset sorted by a metafeature column
        :type UMFj: pandas.core.frame.DataFrame

        :return j or -1: j if the next good instance is found and -1 if not
        :type j: int
        """                   
                     
                     
    j = i
    while j < len(UMFj):
        if UMFj['Algorithm_Perf'][j] == 1:
            return j;
        j += 1;
    
    return -1;

#########################################################################################################################################################################################################

def next_bad_instance(i, UMFj):
                     
        """
        This function is used to determine the next bad instance j (Algorithmic Performance == 0) of a sorted dataset UMFj
        
        :param i: starting index to search for next bad instance
        :type i: int

        :param UMFj: orignal dataset sorted by a metafeature column
        :type UMFj: pandas.core.frame.DataFrame

        :return j or -1: j if the next bad instance is found and -1 if not
        :type j: int
        """      
                     
    j = i
    while j < len(UMFj):
        if UMFj['Algorithm_Perf'][j] == 0:
            return j;
        j += 1;
    
    return -1;

#########################################################################################################################################################################################################

def extend_good_interval(pos, UMFj):
    
        """
        This function is used to extend good intervals by along moving along indexes of UMFj starting from a position pos
        
        :param pos: starting index for extending a good interval
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

def extend_bad_interval(pos, UMFj):
                                       
        """
        This function is used to extend bad intervals by along moving along indexes of UMFj starting from a position pos
        
        :param pos: starting index for extending a bad interval
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

def merge_intervals(GB_ind, GB_val, n, percent_merge):
                     
        """
        This function is used to merge intervals with a maximum gap of n*percent_merge elements
        
        :param GB_ind: index of first and last elements of the original interval
        :type GB_ind: list

        :param GB_val: value of first and last elements of the original interval
        :type GB_val: list
        
        :param n: length of the dataset
        :type n: int
        
        :param percent_merge: percentage to calculate size of gaps between intervals that should be merged 
        :type percent_merge: numpy.float64

        :return GB_ind: index of first and last elements of the new interval
        :type GB_ind: list
        
        :return GB_val: value of first and last elements of the new interval
        :type GB_val: list
        """      
    
    i = 0
    while i < len(GB_ind)-1:
        
        # Here are the interval indexes and values 
        inter1_ind = GB_ind[i]
        inter1_val = GB_val[i]
        inter2_ind = GB_ind[i+1]
        inter2_val = GB_val[i+1]
        
        # if the gap between two intervals is a maximum of n*percent_merge
        if inter2_ind[0] - inter1_ind[1] <= n*percent_merge:
            
            # the new intervals will be the beginning of the first interval with the ending of the  
            # second interval
            newInter_ind = [inter1_ind[0], inter2_ind[1]]
            newInter_val = [inter1_val[0], inter2_val[1]]
            
            #insert new intervals
            GB_ind.insert(i+2, newInter_ind)
            GB_val.insert(i+2, newInter_val)
            
            # remove two prior intervals
            GB_ind.remove(inter1_ind)
            GB_ind.remove(inter2_ind)
            GB_val.remove(inter1_val)
            GB_val.remove(inter2_val)
            
        else: 
            i+=1
            
    return GB_ind, GB_val;

#########################################################################################################################################################################################################

def drop_small_intervals(GB_ind, GB_val, n, percent_drop):
                     
                              
        """
        This function is used to remove intervals with less than n*percent_drop, where n is number of instances of the class
        
        :param GB_ind: index of first and last elements of the original interval
        :type GB_ind: list

        :param GB_val: value of first and last elements of the original interval
        :type GB_val: list
        
        :param n: length of the dataset
        :type n: int
        
        :param percent_drop: percentage to calculate size of intervals that should be dropped 
        :type percent_drop: numpy.float64

        :return GB_ind: index of first and last elements of the new interval
        :type GB_ind: list
        
        :return GB_val: value of first and last elements of the new interval
        :type GB_val: list
        """      
                     
    i = 0
    while i < len(GB_ind):
        if len(range(GB_ind[i][0], GB_ind[i][1]+1)) <= n*percent_drop:
            GB_ind.remove(GB_ind[i])
            GB_val.remove(GB_val[i])
        else: 
            i+=1
    return GB_ind, GB_val;

#########################################################################################################################################################################################################

# This function returns a dictionary of good/bad simple rules for each performance given the good/bad 
# intervals GB of a dataset and the length n of the dataset
def simple_rules(GB, n):

    dict_simple_rules = {}
    
    for performance in GB:
        
        # empty metafeature list (metafeatures can be repeated in this list if there is 
        # a metafeature with more than one interval associated to it)
        metafeature_list = []
        
        # empty interval list for this algorithm
        interval_list = []
        
        # empty interval index list for this algorithm
        interval_ind_list = []
        
        # empty support list (%) which is the percentage of instances that meet the rule
        support_list = []
        
        for metafeature in GB[performance]:
            
            # Get interval values and interval indexes
            int_val = GB[performance][metafeature]['interval_val']
            int_ind = GB[performance][metafeature]['interval_ind']
            
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
                                                               support_list)), columns =['Metafeature', 'Index', 'Interval', 'Support'])
        
    return dict_simple_rules

#########################################################################################################################################################################################################

# This function returns the rules for features that are present in both good and bad intervals for each performance measure
def common_rules(dict_good_rules, dict_bad_rules):
    
    dict_good_rules_common = {}
    dict_bad_rules_common = {}

    for performance in dict_good_rules:
        # features in common
        features_good = list(dict_good_rules[performance]['Metafeature'])
        features_bad = list(dict_bad_rules[performance]['Metafeature'])

        common_features = [feature for feature in features_good if feature in features_bad]

        dict_good_rules_common[performance] = dict_good_rules[performance][dict_good_rules[performance]['Metafeature'].isin(common_features)]
        dict_bad_rules_common[performance] = dict_bad_rules[performance][dict_bad_rules[performance]['Metafeature'].isin(common_features)]
        
    return dict_good_rules_common, dict_bad_rules_common

#########################################################################################################################################################################################################

# This function returns the predictions of performance of the test dataset based on the rules
# generated from the train dataset
def predict(df_performance_test, df_metadata_test, dict_good_rules, dict_bad_rules):

    # Let's define the predicted results dictionary
    dict_pred = {}

    # Let's get the performance names for the current dataset
    performance_names = list(df_performance_test.columns)

    # Let's go through each class (performance)
    for performance in performance_names:

        # Let's create a dataframe with zeros for this performance in this dataset
        dict_pred[performance] = pd.DataFrame(0, index=range(len(df_metadata_test)), columns=["PRD", "NRD"])

        df_good = dict_good_rules[performance]
        df_bad = dict_bad_rules[performance]

        # go through each instance
        for i in range(len(df_metadata_test)):

            # Join in a dataframe the values of a given instance of each metafeature to the interval
            df_instance_meta = pd.DataFrame(df_metadata_test.iloc[i,:])
            df_instance_meta.reset_index(level=0, inplace=True)
            df_instance_meta.columns = ['Metafeature', 'Values']

            df_int_instance_good = df_good.merge(df_instance_meta, on = "Metafeature")
            df_int_instance_bad = df_bad.merge(df_instance_meta, on = "Metafeature")

            # go through each metafeature to check if in interval, if yes break

            for j in range(len(df_int_instance_good)):
                if df_int_instance_good['Interval'][j][0] <= df_int_instance_good['Values'][j] <= df_int_instance_good['Interval'][j][1]:
                    dict_pred[performance]["PRD"][i] = 1
                    break
                else: 
                    continue

            for j in range(len(df_int_instance_bad)):
                if df_int_instance_bad['Interval'][j][0] <= df_int_instance_bad['Values'][j] <= df_int_instance_bad['Interval'][j][1]:
                    dict_pred[performance]["NRD"][i] = 1
                    break
                else: 
                    continue
    
    return dict_pred

#########################################################################################################################################################################################################

# This function saves the percent_merge and percent_drop value combination for each performance measure with
# greatest accuracy score by performing a grid search
def tune_hyper_params(merge_space, drop_space, df_train, df_metadata_train, df_performance_train, df_val, df_performance_val, df_metadata_val, direc, data):
    
    list_merge_value = []
    list_drop_value = []
    list_accuracy = []

    for percent_merge in merge_space:

        for percent_drop in drop_space:

            print(f'percent merge: {percent_merge:.3f}, percent drop: {percent_drop:.2f}')

            # extract the rule intervals using the train data
            
            if data == 'metafeature':
                dict_G, dict_B, = auto_extraction(df_train, df_metadata_train, df_performance_train, percent_drop, percent_merge)
            
            else:
                dict_G, dict_B, = auto_extraction_original(df_train, df_performance_train, percent_drop, percent_merge)
            
            # Good behavior rules
            dict_good_rules = simple_rules(dict_G, len(df_train))

            # Bad behavior rules
            dict_bad_rules = simple_rules(dict_B, len(df_train))

            # Let's keep the features that are present in both good and bad intervals for a given performance measure
            dict_good_rules_common, dict_bad_rules_common = common_rules(dict_good_rules, dict_bad_rules)

            # Let's predict the performance of the validation dataset
            if data == 'metafeature':
                dict_pred = predict(df_performance_val, df_metadata_val, dict_good_rules_common, dict_bad_rules_common)
            else:
                dict_pred = predict(df_performance_val, df_val, dict_good_rules_common, dict_bad_rules_common)

            # Let's evaluate the performance ruleset
            # Let's get the performance names for the current dataset
            performance_names = list(df_performance_val.columns)

            # Here's a temporary list to save the accuracies of each performance metric
            list_accuracy_temp = []

            for performance in performance_names:
                dict_pred[performance].PRD = dict_pred[performance].PRD == 1
                dict_pred[performance].NRD = dict_pred[performance].NRD == 1

                dict_pred[performance]['PRD_not_NRD'] = dict_pred[performance].PRD & ~dict_pred[performance].NRD
                dict_pred[performance]['NRD_not_PRD'] = ~dict_pred[performance].PRD & dict_pred[performance].NRD

                # Which rules are used to classify Good and Bad instances
                mask1 = dict_pred[performance].PRD_not_NRD == True
                mask2 = dict_pred[performance].NRD_not_PRD == True

                dict_pred[performance]['Final'] = [-1] * len(df_metadata_val)

                dict_pred[performance].loc[mask2, ('Final')] = 0
                dict_pred[performance].loc[mask1, ('Final')] = 1

                # accuracy
                acc = accuracy_score(df_performance_val[performance], dict_pred[performance].Final)

                # Let's append to the temporary list
                list_accuracy_temp.append(acc)

            # add the accuracies to the final accuracy list
            list_accuracy.append(list_accuracy_temp)

            # add percent_merge to list
            list_merge_value.append(percent_merge)

            # add percent_drop to list
            list_drop_value.append(percent_drop)
    
    
    # Get the best accuracies and save their respective merge and drop percentages
    df_accuracy = pd.DataFrame(list_accuracy, columns=performance_names)

    list_best_merge = []
    list_best_drop = []

    for performance in performance_names:
        best_index = df_accuracy.index[df_accuracy[performance] == max(df_accuracy[performance])].tolist()[0]
        list_best_merge.append(list_merge_value[best_index])
        list_best_drop.append(list_drop_value[best_index])
    
    df_best = pd.DataFrame([list_best_merge, list_best_drop], columns=performance_names, index=['merge', 'drop'])
    
    if data == 'metafeature':
        df_best.to_csv(f'{direc}/best_hyperparameters.csv', index=False)
    else: 
        df_best.to_csv(f'{direc}/best_hyperparameters_original.csv', index=False)
    
    return df_accuracy
#########################################################################################################################################################################################################




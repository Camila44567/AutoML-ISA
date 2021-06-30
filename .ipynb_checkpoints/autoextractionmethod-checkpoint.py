import operator
import pandas as pd
from collections import defaultdict
from IPython.display import display_html
from itertools import chain,cycle

# This function is used to get intervals of Good and Bad instances
# for each dataset
def getIntervals(datasets_dict):
    
    # list of dataset names
    dataset_names = list(datasets_dict['original_features'].keys())

    # Let's define the good and bad interval dictionary
    G = {}
    B = {}


    # Let's go through each dataset
    # We'll be evaluating the domains of competence of each algorithm within each dataset
    for dataset in dataset_names:

        # Let's create an empty dictionary for this dataset
        G[dataset] = {}
        B[dataset] = {}

        # Let's get the algorithm names for the current dataset
        algorithm_names = list(datasets_dict['algorithm_bin'][dataset].columns)

        # Let's go through each class (algorithm performance)
        for algorithm in algorithm_names:

            # Let's create an empty dictionary for this algorithm in this dataset
            G[dataset][algorithm] = {}
            B[dataset][algorithm] = {}

            # Let's get the meta feature names for the current dataset
            meta_feature_names = list(datasets_dict['meta_features'][dataset].columns)


            # let's go through each column of the meta features (MFj)
            for metafeature in meta_feature_names:

                # Let's create an empty dictionary for this metafeature 
                # in this algorithm in this dataset
                G[dataset][algorithm][metafeature] = {}
                B[dataset][algorithm][metafeature] = {}


                # let's create an auxiliary empty list of good and bad interval indexes and values 
                G_int_ind = []
                G_int_val = []
                B_int_ind = []
                B_int_val = []


                # add MFj that will be used for sorting to dataset and 
                # algorithm performance from algorithm_bin to define good/bad elements 
                U = datasets_dict['original_features'][dataset].assign(
                    MFj = datasets_dict['meta_features'][dataset][metafeature], 
                    Algorithm_Perf = datasets_dict['algorithm_bin'][dataset][algorithm])

                # sort list U by each meta feature MFj
                UMFj = U.sort_values(by=['MFj'])
                UMFj.reset_index(inplace=True)

                # let's search for good behavior intervals
                i = 0
                pos = 0
                while i < len(UMFj)-1 and pos != -1:

                    # position of the next good instance
                    pos = nextGoodInstance(i, UMFj)

                    if pos != -1:
                        V_ind, V_val, i = extendGoodInterval(pos, UMFj)
                        G_int_ind.append(V_ind)
                        G_int_val.append(V_val)

                # let's search for bad behavior intervals
                i = 0
                pos = 0
                while i < len(UMFj)-1 and pos != -1:

                    # position of the next bad instance
                    pos = nextBadInstance(i, UMFj)
                    if pos != -1:
                        V_ind, V_val, i = extendBadInterval(pos, UMFj)
                        B_int_ind.append(V_ind)
                        B_int_val.append(V_val)

                # Let's merge and filter the intervals if necessary
                G_int_ind, G_int_val = mergeIntervals(G_int_ind, G_int_val)
                B_int_ind, B_int_val = mergeIntervals(B_int_ind, B_int_val)

                G_int_ind, G_int_val = dropSmallIntervals(G_int_ind, G_int_val, len(UMFj))
                B_int_ind, B_int_val = dropSmallIntervals(B_int_ind, B_int_val, len(UMFj))

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

                # Add G_aux and B_aux to dictionaries according to the current 
                # dataset, algorithm, metafeature and interval        
                G[dataset][algorithm][metafeature]['interval_ind'] = G_int_ind_aux
                G[dataset][algorithm][metafeature]['interval_val'] = G_int_val_aux
                B[dataset][algorithm][metafeature]['interval_ind'] = B_int_ind_aux
                B[dataset][algorithm][metafeature]['interval_val'] = B_int_val_aux
    
    return G, B


# This function is used to determine the next good instance j (Algorithmic Performance == 1)
# of a sorted dataset UMFj from a starting point i
# if j is not found, it returns -1
def nextGoodInstance(i, UMFj):
    j = i
    while j < len(UMFj):
        if UMFj['Algorithm_Perf'][j] == 1:
            return j;
        j += 1;
    
    return -1;
    
# This function is used to determine the next bad instance j (Algorithmic Performance == 0)
# of a sorted dataset UMFj from a starting point i
# if j is not found, it returns -1
def nextBadInstance(i, UMFj):
    j = i
    while j < len(UMFj):
        if UMFj['Algorithm_Perf'][j] == 0:
            return j;
        j += 1;
    
    return -1;

# This function is used to extend good intervals by along moving along indexes of
# UMFj starting from a position pos
# it returns the indexes and values of the first and last element of the interval, and
# the limit where the algorithm stopped
def extendGoodInterval(pos, UMFj):
    limit = pos
    end = False
    while end == False and UMFj['Algorithm_Perf'][limit] == 1:
        limit += 1;
        if limit > len(UMFj)-1:
            limit = len(UMFj)-1
            end = True
    
    return [pos, limit], [round(UMFj['MFj'][pos], 2), round(UMFj['MFj'][limit], 2)], limit;

# This function is used to extend bad intervals by along moving along indexes of
# UMFj starting from a position pos
# it returns the indexes and values of the first and last element of the interval, and
# the limit where the algorithm stopped
def extendBadInterval(pos, UMFj):
    limit = pos
    end = False
    while end == False and UMFj['Algorithm_Perf'][limit] == 0:
        limit += 1;
        if limit > len(UMFj)-1:
            limit = len(UMFj)-1
            end = True
    
    return [pos, limit], [round(UMFj['MFj'][pos], 2), round(UMFj['MFj'][limit], 2)], limit;


# This function is used to merge intervals with a maximum gap of 5 elements
def mergeIntervals(GB_ind, GB_val):
    i = 0
    while i < len(GB_ind)-1:
        
        # Here are the interval indexes and values 
        inter1_ind = GB_ind[i]
        inter1_val = GB_val[i]
        inter2_ind = GB_ind[i+1]
        inter2_val = GB_val[i+1]
        
        # if the gap between two intervals is a maximum of 5
        if inter2_ind[0] - inter1_ind[1] <= 5:
            
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

# This function is used to remove intervals with less than n*0.15, where n is number of instances
# in the dataset
def dropSmallIntervals(GB_ind, GB_val, n):
    i = 0
    while i < len(GB_ind):
        if len(range(GB_ind[i][0], GB_ind[i][1]+1)) <= n*0.15:
            GB_ind.remove(GB_ind[i])
            GB_val.remove(GB_val[i])
        else: 
            i+=1
    return GB_ind, GB_val;

# This function returns a dictionary of good/bad simple rules for each algorithm given the good/bad 
# intervals GB of a dataset and the length n of the dataset
def simpleRules(GB, n):

    Simple_Rules = {}
    
    for algorithm in GB:
        
        # empty metafeature list (metafeatures can be repeated in this list if there is 
        # a matafeature with more than one interval associated to it)
        metafeature_list = []
        
        # empty interval list for this algorithm
        interval_list = []
        
        # empty interval index list for this algorithm
        interval_ind_list = []
        
        
        # empty support list (%) which is the percentage of instances that meet the rule
        support_list = []
        
        
        
        for metafeature in GB[algorithm]:
            
            for int_val in GB[algorithm][metafeature]['interval_val']:
                
                interval_list.append(int_val)
                metafeature_list.append(metafeature.replace("feature_", ""))
            
            for int_ind in GB[algorithm][metafeature]['interval_ind']:
                
                interval_ind_list.append(int_ind)
                support_list.append(round((len(range(int_ind[0], int_ind[1]+1))/n), 2))
                
        Simple_Rules[algorithm] = pd.DataFrame(list(zip(metafeature_list, interval_ind_list, interval_list, support_list)), columns =['Metafeature', 'Index', 'Interval', 'Support'])
        
    return(Simple_Rules)

def display_side_by_side(*args,titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)
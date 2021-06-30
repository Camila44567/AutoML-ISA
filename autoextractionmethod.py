import operator
import pandas as pd
from IPython.display import display_html
from itertools import chain,cycle


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
import pandas as pd
import os

def create_sets(direc):
    
    datasets_dict = {
    'original_features' : {},
        'meta_features' : {},
        'algorithm_bin' : {},
            'beta_easy' : {} } # empty nested dict that will receive our newly created datasets

    for d in direc:
    
        # read original features
        of = pd.read_csv(f'{d}/data.csv')
        # Drop last column (class)
        of = of.iloc[: , :-1]

        # read metafeature set
        md = pd.read_csv(f'{d}/metadata.csv')
        md = md.drop('instances', axis=1) # remove instance number column
        
        #remove algorithms from metafeature set
        algos = ['algo_bagging', 'algo_gradient_boosting', 'algo_logistic_regression', 
             'algo_mlp', 'algo_random_forest', 'algo_svc_linear', 'algo_svc_rbf']

        for a in algos:
            del md[a]
        
        # read algorithm bin
        ab = pd.read_csv(f'{d}/algorithm_bin.csv')
        ab = ab.drop('Row', axis=1) # remove instance number column

        # read beta easy
        be = pd.read_csv(f'{d}/beta_easy.csv')
        
        # Add data to dictionary
        key = os.path.basename(d)
        datasets_dict['original_features'][key] = of
        datasets_dict['meta_features'][key] = md
        datasets_dict['algorithm_bin'][key] = ab
        datasets_dict['beta_easy'][key] = be['IsBetaEasy']
    
    return datasets_dict
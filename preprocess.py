import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_sets(direc):
    
    for d in direc:
    
        # read original features
        X = pd.read_csv(f'{d}/data.csv')
        y = X.iloc[: , -1]

        # Drop last column (class)
        X = X.iloc[: , :-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        # Rejoin X and y 
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)
        
        # Create train and test folders in the directory
        try:
            os.makedirs(f'{d}/Train')
        except:
            print("There is already a Train folder")
        try:
            os.makedirs(f'{d}/Test')    
        except:
            print("There is already a Test folder")
            
        # Write as csv the train and test dataset
        train.to_csv(f'{d}/Train/data.csv', index=False)
        test.to_csv(f'{d}/Test/data.csv', index=False)
    
def create_sets(direc, sub):
    
    datasets_dict = {
    'original_features' : {},
        'meta_features' : {},
        'algorithm_bin' : {},
            'beta_easy' : {} } # empty nested dict that will receive our newly created datasets

    for d in direc:
    
        # read original features
        of = pd.read_csv(f'{d}/{sub}/data.csv')
        # Drop last column (class)
        of = of.iloc[: , :-1]

        # read metafeature set
        md = pd.read_csv(f'{d}/{sub}/metadata.csv')
        md = md.drop('instances', axis=1) # remove instance number column
        
        #remove algorithms from metafeature set
        algos = ['algo_bagging', 'algo_gradient_boosting', 'algo_logistic_regression', 
             'algo_mlp', 'algo_random_forest', 'algo_svc_linear', 'algo_svc_rbf']

        for a in algos:
            del md[a]
        
        # read algorithm bin
        ab = pd.read_csv(f'{d}/{sub}/algorithm_bin.csv')
        ab = ab.drop('Row', axis=1) # remove instance number column

        # read beta easy
        be = pd.read_csv(f'{d}/{sub}/beta_easy.csv')
        
        # Add data to dictionary
        key = os.path.basename(d)
        datasets_dict['original_features'][key] = of
        datasets_dict['meta_features'][key] = md
        datasets_dict['algorithm_bin'][key] = ab
        datasets_dict['beta_easy'][key] = be['IsBetaEasy']
    
    return datasets_dict
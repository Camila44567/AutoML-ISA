import pandas as pd
import os
from sklearn.model_selection import train_test_split
import time
import os
import sys
import logging
from pathlib import Path
from pyhard.context import Configuration, Workspace
from pyhard import integrator, formatter
from pyhard.feature_selection import featfilt
from inspect import signature, Parameter
from pyispace import train_is
import json
from sklearn_extra.cluster import KMedoids
import numpy as np
import random

#########################################################################################################################################################################################################

def split_sets(X, y):
    
    # random stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)

    # Rejoin X and y 
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    
    # Reset index
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    return train, test

#########################################################################################################################################################################################################

def create_isa_datasets(direc):
    
    file = f"{direc}/data.csv"
    
    # read original features
    df_original = pd.read_csv(file)
    
    kwargs = {'rootdir': '/home/camila/Documents/Faculdade/Projeto-Mestrado/AutoML-ISA', 
     'datafile': '/home/camila/Documents/Faculdade/Projeto-Mestrado/AutoML-ISA/data.csv', 
     'problem': 'classification', 'seed': 0, 'isa_engine': 'python', 'n_folds': 5, 'n_iter': 1,
     'metric': 'logloss', 'perf_threshold': 'auto', 'feat_select': 'x', 'max_n_features': 10, 
     'method': 'mrmr', 'var_filter': True, 'var_threshold': 0, 'hyper_param_optm': True, 
     'hpo_evals': 20, 'hpo_timeout': 90, 'adjust_rotation': True, 'ih_threshold': 0.4, 
     'ih_purity': 0.55, 'measures_list': ['kDN', 'DCP', 'TD_P', 'TD_U', 'CL', 'CLD', 'N1', 'N2', 'LSC', 'LSR', 'Harmfulness', 'Usefulness', 'F1'],
     'algo_list': ['svc_linear', 'svc_rbf', 'random_forest', 'gradient_boosting', 'bagging', 'logistic_regression', 'mlp'], 
     'parameters': {'random_forest': {'n_jobs': -1}, 'bagging': {'n_jobs': -1}, 'dummy': {'strategy': 'prior'}}
    }

    # Get metadata and instance hardness for the dataset
    df_metadata = integrator.build_metadata(data=df_original, return_ih=False,
                                                            verbose=False, **kwargs)
    
    
    # select features
    n_feat_cols = len(df_metadata.filter(regex='^feature_').columns)

    sig = signature(featfilt)
    param_dict = {param.name: kwargs[param.name] for param in sig.parameters.values()
                  if param.kind == param.POSITIONAL_OR_KEYWORD and param.default != Parameter.empty and param.name in kwargs}
    
    selected, df_metadata = featfilt(df_metadata, **param_dict)
    
    # Run ISA
    
    rootdir_path = Path(kwargs['rootdir'])
    opts_path = rootdir_path / 'options.json'

    if opts_path.is_file():
        with open(str(opts_path)) as f:
            opts = json.load(f)

    out = train_is(df_metadata, opts, rotation_adjust=True)

    # get the algorithm_bin and beta_easy for the dataset
    
    output_idx_name = 'Row'
    idx = pd.Index(list(range(1, out.pilot.Z.shape[0] + 1)), name=output_idx_name)
    
    # algorithm_bin
    df_algorithm_bin = pd.DataFrame(out.data.Ybin.astype(int), index=idx, columns=out.data.algolabels)

    # beta_easy
    df_beta_easy = pd.DataFrame(out.data.beta.astype(int), index=idx, columns=['IsBetaEasy'])
    
    
    # Create train, validation and test datasets
    
    # Class
    df_y = df_original.iloc[: , -1]

    # Drop last column (class)
    df_X = df_original.iloc[: , :-1]
    
    df_train_aux, df_test = split_sets(df_X, df_y)
    
    df_train, df_val = split_sets(df_train_aux.iloc[: , :-1], df_train_aux.iloc[: , -1])
    
    # get indices
    train_idx = df_train.index
    test_idx = df_test.index
    val_idx = df_val.index
    
    
    
    # Write original, metadata, algorithm_bin and beta_easy for train, test and validation data to .csv
    
    df_original.iloc[train_idx].to_csv(f'{direc}/train.csv', index=False)
    df_original.iloc[test_idx].to_csv(f'{direc}/test.csv', index=False)
    df_original.iloc[val_idx].to_csv(f'{direc}/val.csv', index=False)

    df_metadata.iloc[train_idx].to_csv(f'{direc}/metadata_train.csv', index=False)
    df_metadata.iloc[test_idx].to_csv(f'{direc}/metadata_test.csv', index=False)
    df_metadata.iloc[val_idx].to_csv(f'{direc}/metadata_val.csv', index=False)

    df_algorithm_bin.iloc[train_idx].to_csv(f'{direc}/algorithm_bin_train.csv', index=False)
    df_algorithm_bin.iloc[test_idx].to_csv(f'{direc}/algorithm_bin_test.csv', index=False)
    df_algorithm_bin.iloc[val_idx].to_csv(f'{direc}/algorithm_bin_val.csv', index=False)

    df_beta_easy.iloc[train_idx].to_csv(f'{direc}/beta_easy_train.csv', index=False)
    df_beta_easy.iloc[test_idx].to_csv(f'{direc}/beta_easy_test.csv', index=False)
    df_beta_easy.iloc[val_idx].to_csv(f'{direc}/beta_easy_val.csv', index=False)

############################################################################################
    

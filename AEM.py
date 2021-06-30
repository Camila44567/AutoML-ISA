import joindata as jd
import os

dataset_names = next(os.walk('/home/camila/Documents/Faculdade/Projeto-Mestrado/Dados/Reais'))[1]
print("Dataset names:")
print(dataset_names)

# Directories with data for original features, metadata, algorithm bin and beta easy
direc = ["/home/camila/Documents/Faculdade/Projeto-Mestrado/Dados/Reais/credit-g",
         "/home/camila/Documents/Faculdade/Projeto-Mestrado/Dados/Reais/compas",
         "/home/camila/Documents/Faculdade/Projeto-Mestrado/Dados/Reais/glass",
         "/home/camila/Documents/Faculdade/Projeto-Mestrado/Dados/Reais/iris",
         "/home/camila/Documents/Faculdade/Projeto-Mestrado/Dados/Reais/pima_diabetes"
        ]

datasets = [] # empty list that will receive our newly created datasets

for d in direc:
    datasets.append(jd.create_sets(d))

    
#remove algorithms from feature set
algos = ['algo_bagging', 'algo_gradient_boosting', 'algo_logistic_regression', 
         'algo_mlp', 'algo_random_forest', 'algo_svc_linear', 'algo_svc_rbf']

for i in range(len(datasets)):
    for a in algos:
        del datasets[i][1][a]

original_feature_names = []
meta_feature_names = []
for data in datasets:
    original_feature_names.append(data[0].columns)
    meta_feature_names.append(data[1].columns)
    

class_sets = datasets[0][2].columns
print("\n")
print("Class set names:")
print(class_sets)

print("\n")
print("Meta feature names:")
print(meta_feature_names)

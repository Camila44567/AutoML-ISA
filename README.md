# IH-AEM
_Instance Hardness Automatic Extraction Method_


## Getting Started



For _preprocessing_ the following steps are taken:

1. Calculate the _hardness measures_;

2. Evaluate classification performance at instance level for each algorithm;

3. Select the most relevant hardness measures with respect to the instance classification error;

4. Build metadata, performance and easiness files (metadata, algorithm_bin and beta_easy) divided into train, validation and test;



For the _Automatic Extraction Method_ the following steps are taken:

1. Order instances and performance measures by each metafeature;

2. Find continuous intervals of easy and hard instances;

3. Merge or drop intervals according to the hyperparameters percent_merge and percent_drop, which are previously tuned;

4. With the meta-feature intervals create rules for easy and hard instances;

5. Use the rules to classify test instances as easy or hard and evaluate the F1 score;



### Input file

Please follow the guidelines below:

* Only `csv` files are accepted

* The dataset should be in the format `(n_instances, n_features)`

* **Do not** include any index column. Instances will be indexed in order, starting from **1**

* **The last column** must contain the classes of the instances

* Categorical features should be handled previously

## References

_Base_

1. Michael R. Smith, Tony Martinez, and Christophe Giraud-Carrier. 2014. __An instance level analysis of data complexity__. Mach. Learn. 95, 2 (May 2014), 225–256.

1. Luengo J, Herrera F. _An  automatic  extraction  method  of  the  domains  of  competence  for learning classifiers using data complexity measures_. Knowledge and Information Systems. 2013 Oct;42(1):147–180. Available from:https://doi.org/10.1007/s10115-013-0700-4.
